from __future__ import annotations

import asyncio
import gc
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import NamedTuple

import torch
from torch import nn

from .extractor import Extractor
from .hooks import decoder_of, install, repair_time_step_limits
from .transport import SLOTS, BufferPool, capture_seconds, finish_copy, memory_peaks, publish, timeline
from .types import ActivationBatch, DeviceActivationBatch, validate_tokens


def _cuda_device_index(device: torch.device) -> int:
    """Resolve an unindexed CUDA device in the calling thread."""
    return device.index if device.index is not None else torch.cuda.current_device()


class Capture(NamedTuple):
    values: dict
    copied: object
    forward_events: object
    copy_events: object
    capture_events: tuple


class HFExtractor(Extractor):
    """Eager PyTorch reference; accepts a model ID or a caller-owned nn.Module."""

    def __init__(
        self,
        model,
        sites,
        *,
        revision=None,
        device="cuda",
        dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=False,
        diagnostics=False,
        publication_threads=1,
    ):
        super().__init__(sites, diagnostics, publication_threads)
        self.owns_model = not isinstance(model, nn.Module)
        started = time.perf_counter()
        if self.owns_model:
            from transformers import AutoModelForCausalLM

            model = AutoModelForCausalLM.from_pretrained(
                model,
                revision=revision,
                torch_dtype=dtype,
                device_map=device,
                trust_remote_code=trust_remote_code,
                local_files_only=local_files_only,
            )
        repair_time_step_limits(model)
        self.model_load_seconds = time.perf_counter() - started
        self.model = model
        self.device = torch.device(device)
        self.captured, self.handles = {}, []
        self.capture_events = []
        self.model_forward_seconds = 0.0
        self.cpu_copy_seconds = 0.0
        self.pool = BufferPool()
        self.producer = ThreadPoolExecutor(max_workers=1)
        self.writer = ThreadPoolExecutor(max_workers=publication_threads)
        self.metrics_lock = threading.Lock()
        self.copy_worker = ThreadPoolExecutor(max_workers=1)
        self.capture_lock = threading.Lock()
        self.stream_lock = threading.Lock()
        self.copy_stream = torch.cuda.Stream(device=self.device) if self.device.type == "cuda" else None
        self.origin = None
        if self.copy_stream is not None:
            self.origin = torch.cuda.Event(enable_timing=True)
            self.origin.record()
            torch.cuda.synchronize(self.device)
        self.origin_cpu = time.perf_counter()
        started = time.perf_counter()
        try:
            self.handles, self.topology = install(
                model, self.sites, self.captured, capture_events=self.capture_events if diagnostics else None
            )
            self.hook_install_seconds = time.perf_counter() - started
        except Exception:
            self.close()
            raise

    def extract(self, token_ids):
        self._require_open()
        tokens = validate_tokens(token_ids)
        if not self.stream_lock.acquire(blocking=False):
            raise RuntimeError("Another extraction stream is active")
        try:
            slot = self.pool.acquire()
            try:
                capture = self._capture(tokens, slot)
                self._finish(capture)
                return ActivationBatch(tokens, {site.key: capture.values[site.key].clone() for site in self.sites})
            finally:
                self.pool.release(slot)
        finally:
            self.stream_lock.release()

    def extract_device(self, token_ids):
        """Capture frozen-backbone activations without CPU copies or vocabulary logits.

        The returned detached CUDA tensors own their storage. Callers retain
        them until asynchronous peer transfers and local consumers complete.
        """
        self._require_open()
        tokens = validate_tokens(token_ids)
        if not self.stream_lock.acquire(blocking=False):
            raise RuntimeError("Another extraction stream is active")
        try:
            with self._evaluating() as decoder:
                with torch.inference_mode():
                    decoder(input_ids=tokens.to(self.device), use_cache=False)
                return DeviceActivationBatch(
                    tokens,
                    {
                        site.key: self.captured[site.key].reshape(-1, self.captured[site.key].shape[-1]).clone()
                        for site in self.sites
                    },
                )
        finally:
            self.stream_lock.release()

    @contextmanager
    def _evaluating(self):
        """Hold the capture lock in eval mode; yield the decoder, skipping vocabulary logits."""
        with self.capture_lock:
            was_training = self.model.training
            self.captured.clear()
            try:
                self.model.eval()
                yield decoder_of(self.model)
            finally:
                self.captured.clear()
                self.model.train(was_training)

    def _capture(self, tokens, slot):
        copy_events = None
        with self._evaluating() as decoder:
            self.capture_events.clear()
            with torch.inference_mode():
                inputs = tokens.to(self.device)
                events = None
                started = time.perf_counter()
                if self.copy_stream is not None:
                    torch.cuda.set_device(_cuda_device_index(self.device))
                    events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                    events[0].record()
                decoder(input_ids=inputs, use_cache=False)
                if events:
                    events[1].record()
                else:
                    self.model_forward_seconds += time.perf_counter() - started
            values = {"token_ids": tokens}
            sources = dict(self.captured)
            for site in self.sites:
                value = sources[site.key]
                values[site.key] = self.pool.tensor(
                    slot, site.key, (tokens.numel(), value.shape[-1]), value.dtype, pinned=self.copy_stream is not None
                )
                self.topology[site.key]["width"] = value.shape[-1]
            if self.copy_stream is not None:
                copy_start, copy_end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                with torch.cuda.stream(self.copy_stream):
                    # Event dependency works for callers using any compute stream.
                    self.copy_stream.wait_event(events[1])
                    copy_start.record()
                    for key, source in sources.items():
                        values[key].copy_(source.reshape(values[key].shape), non_blocking=True)
                        source.record_stream(self.copy_stream)
                    copy_end.record()
                copied = self.copy_worker.submit(finish_copy, copy_start, copy_end, sources)
                copy_events = (copy_start, copy_end)
                self._fold_memory(memory_peaks(device=self.device))
            else:
                for key, source in sources.items():
                    values[key].copy_(source.reshape(values[key].shape))
                copied = self.copy_worker.submit(finish_copy, None, None, sources)
            self.actual_forward_batches.append(len(tokens))
            return Capture(values, copied, events, copy_events, tuple(self.capture_events))

    def _finish(self, capture):
        seconds = capture.copied.result()
        with self.metrics_lock:
            self.cpu_copy_seconds += seconds
            if capture.forward_events:
                events = capture.forward_events
                self.model_forward_seconds += events[0].elapsed_time(events[1]) / 1000

    def _publish(self, tokens, slot, capture, path, index):
        try:
            self._finish(capture)
            write_start = time.perf_counter() - self.origin_cpu
            receipt = publish(capture.values, path)
            with self.metrics_lock:
                self.worker_write_seconds += receipt["write_seconds"]
                self.validation_seconds += receipt["validation_seconds"]
                self.serialization_seconds += receipt["serialization_seconds"]
            receipt.update(index=index, tokens=tokens.numel())
            receipt["capture_seconds"] = capture_seconds(capture.capture_events) if self.diagnostics else None
            receipt["timeline"] = (
                timeline(self.origin, (("forward", [capture.forward_events]), ("D2H", [capture.copy_events])))
                if capture.forward_events
                else []
            )
            receipt["timeline"].append(
                {"name": "write", "start": write_start, "end": time.perf_counter() - self.origin_cpu}
            )
            return receipt
        finally:
            self.pool.release(slot)

    async def extract_async(self, token_ids):
        return await asyncio.get_running_loop().run_in_executor(self.producer, self.extract, token_ids)

    async def stream_to_disk(self, token_batches, output_dir):
        self._require_open()
        if not self.stream_lock.acquire(blocking=False):
            raise RuntimeError("Another extraction stream is active")
        pending = []
        loop = asyncio.get_running_loop()
        try:
            for index, batch in enumerate(token_batches):
                tokens = validate_tokens(batch)
                if len(pending) >= SLOTS:
                    started = time.perf_counter()
                    receipt = await asyncio.shield(asyncio.wrap_future(pending[0]))
                    pending.pop(0)
                    yield receipt
                    self.backpressure_seconds += time.perf_counter() - started
                # Draining above released the oldest slot, so one is always free here.
                slot = self.pool.acquire()
                try:
                    # Shield CUDA submissions: cancellation must not release a live slot.
                    job = loop.run_in_executor(self.producer, self._capture, tokens, slot)
                    try:
                        capture = await asyncio.shield(job)
                    except asyncio.CancelledError:
                        capture = await job
                        await asyncio.to_thread(self._finish, capture)
                        raise
                    pending.append(
                        self.writer.submit(
                            self._publish, tokens, slot, capture, self.shard_path(output_dir, index), index
                        )
                    )
                except BaseException:
                    self.pool.release(slot)
                    raise
            started = time.perf_counter()
            while pending:
                receipt = await asyncio.shield(asyncio.wrap_future(pending[0]))
                pending.pop(0)
                yield receipt
            self.final_drain_seconds += time.perf_counter() - started
        finally:
            unwinding = sys.exc_info()[0] is not None
            try:
                # Drain every writer even when the first one failed.
                results = await asyncio.shield(
                    asyncio.gather(*(asyncio.wrap_future(future) for future in pending), return_exceptions=True)
                )
                if not unwinding:
                    for result in results:
                        if isinstance(result, BaseException):
                            raise result
            finally:
                self.stream_lock.release()

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.producer.shutdown(wait=True)
        self.copy_worker.shutdown(wait=True)
        self.writer.shutdown(wait=True)
        self.pool.reset()
        for handle in self.handles:
            handle.remove()
        self.captured.clear()
        if self.owns_model:
            self.model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
