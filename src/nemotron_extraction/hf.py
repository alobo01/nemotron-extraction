from __future__ import annotations
import gc
import asyncio
import threading
import time
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import torch
from torch import nn
from .hooks import install, repair_time_step_limits
from .transport import BufferPool, MEMORY_KEYS, capture_seconds, finish_copy, memory_peaks, publish, timeline
from .types import ActivationBatch, DeviceActivationBatch, validate_publication_threads, validate_sites, validate_tokens


def _cuda_device_index(device: torch.device) -> int:
    """Resolve an unindexed CUDA device in the calling thread."""
    return device.index if device.index is not None else torch.cuda.current_device()


class HFExtractor:
    """Eager PyTorch reference; accepts a model ID or a caller-owned nn.Module."""
    def __init__(self, model, sites, *, revision=None, device="cuda", dtype=torch.bfloat16,
                 trust_remote_code=True, local_files_only=False, gpu_memory_utilization=0.96,
                 diagnostics=False, publication_threads=1):
        self.diagnostics = diagnostics
        self.publication_threads = validate_publication_threads(publication_threads)
        self.sites = validate_sites(sites)
        self.closed = False
        self.owns_model = not isinstance(model, nn.Module)
        started = time.perf_counter()
        if self.owns_model:
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(
                model, revision=revision, torch_dtype=dtype, device_map=device,
                trust_remote_code=trust_remote_code, local_files_only=local_files_only)
        repair_time_step_limits(model)
        self.model_load_seconds = time.perf_counter() - started
        self.model = model
        self.device = torch.device(device)
        self.captured, self.handles = {}, []
        self.capture_events = []
        self.model_forward_seconds = 0.0
        self.cpu_copy_seconds = 0.0
        self.gpu_memory_utilization = gpu_memory_utilization
        self.pool = BufferPool()
        self.producer = ThreadPoolExecutor(max_workers=1)
        self.writer = ThreadPoolExecutor(max_workers=publication_threads)
        self.metrics_lock = threading.Lock()
        self.validation_seconds = self.serialization_seconds = 0.0
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
        self.worker_write_seconds = self.backpressure_seconds = self.final_drain_seconds = 0.0
        self.driver_read_seconds = 0.0
        self.actual_forward_batches = []
        self.peak_gpu_memory_bytes = self.peak_gpu_reserved_bytes = self.peak_device_memory_bytes = 0
        self.total_device_memory_bytes = 0
        self.peak_device_memory_fraction = 0.0
        started = time.perf_counter()
        try:
            self.handles, self.topology = install(model, self.sites, self.captured,
                capture_events=self.capture_events if diagnostics else None)
            self.hook_install_seconds = time.perf_counter() - started
        except Exception:
            self.close()
            raise

    def extract(self, token_ids):
        if self.closed:
            raise RuntimeError("Extractor is closed")
        tokens = validate_tokens(token_ids)
        if not self.stream_lock.acquire(blocking=False):
            raise RuntimeError("Another extraction stream is active")
        try:
            slot = self.pool.acquire()
            try:
                tensors, copied, events = self._capture(tokens, slot)
                self._finish(copied, events)
                return ActivationBatch(tokens, {site.key: tensors[site.key].clone() for site in self.sites})
            finally:
                self.pool.release(slot)
        finally:
            self.stream_lock.release()

    def extract_device(self, token_ids):
        """Capture frozen-backbone activations without CPU copies or vocabulary logits.

        The returned detached CUDA tensors own their storage. Callers retain
        them until asynchronous peer transfers and local consumers complete.
        """
        if self.closed:
            raise RuntimeError("Extractor is closed")
        tokens = validate_tokens(token_ids)
        if not self.stream_lock.acquire(blocking=False):
            raise RuntimeError("Another extraction stream is active")
        try:
            with self.capture_lock:
                self.captured.clear()
                was_training = self.model.training
                self.model.eval()
                try:
                    decoder = getattr(self.model, "backbone", None)
                    if decoder is None:
                        decoder = getattr(self.model, "model", self.model)
                    with torch.inference_mode():
                        decoder(input_ids=tokens.to(self.device), use_cache=False)
                    return DeviceActivationBatch(tokens, {
                        site.key: self.captured[site.key].reshape(-1, self.captured[site.key].shape[-1]).clone()
                        for site in self.sites})
                finally:
                    self.captured.clear()
                    self.model.train(was_training)
        finally:
            self.stream_lock.release()

    def _capture(self, tokens, slot):
        with self.capture_lock:
            return self._capture_locked(tokens, slot)

    def _capture_locked(self, tokens, slot):
        self.captured.clear()
        self.capture_events.clear()
        was_training = self.model.training
        try:
            self.model.eval()
            with torch.inference_mode():
                decoder = getattr(self.model, "backbone", None)
                if decoder is None:
                    decoder = getattr(self.model, "model", self.model)
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
                values[site.key] = self.pool.tensor(slot, site.key, (tokens.numel(), value.shape[-1]),
                                                   value.dtype, pinned=self.copy_stream is not None)
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
                copied.copy_events = (copy_start, copy_end)
                self.__dict__.update(memory_peaks({key: getattr(self, key) for key in MEMORY_KEYS}, self.device))
            else:
                for key, source in sources.items():
                    values[key].copy_(source.reshape(values[key].shape))
                copied = self.copy_worker.submit(finish_copy, None, None, sources)
            self.actual_forward_batches.append(len(tokens))
            copied.capture_events = tuple(self.capture_events)
            return values, copied, events
        finally:
            self.captured.clear()
            self.model.train(was_training)

    def _finish(self, copied, events):
        seconds = copied.result()
        with self.metrics_lock:
            self.cpu_copy_seconds += seconds
            if events:
                self.model_forward_seconds += events[0].elapsed_time(events[1]) / 1000

    def _publish(self, tokens, slot, tensors, copied, events, path, index):
        try:
            self._finish(copied, events)
            write_start = time.perf_counter() - self.origin_cpu
            receipt = publish(tensors, path)
            with self.metrics_lock:
                self.worker_write_seconds += receipt["write_seconds"]
                self.validation_seconds += receipt["validation_seconds"]
                self.serialization_seconds += receipt["serialization_seconds"]
            receipt.update(index=index, tokens=tokens.numel())
            receipt["capture_seconds"] = capture_seconds(copied.capture_events) if self.diagnostics else None
            receipt["timeline"] = (timeline(self.origin, (("forward", [events]), ("D2H", [copied.copy_events])))
                                   if events else [])
            receipt["timeline"].append({"name": "write", "start": write_start,
                                        "end": time.perf_counter() - self.origin_cpu})
            return receipt
        finally:
            self.pool.release(slot)

    async def extract_async(self, token_ids):
        return await asyncio.get_running_loop().run_in_executor(self.producer, self.extract, token_ids)

    async def stream_to_disk(self, token_batches, output_dir):
        if self.closed:
            raise RuntimeError("Extractor is closed")
        if not self.stream_lock.acquire(blocking=False):
            raise RuntimeError("Another extraction stream is active")
        pending = []
        loop = asyncio.get_running_loop()
        try:
            for index, batch in enumerate(token_batches):
                tokens = validate_tokens(batch)
                if len(pending) >= 2:
                    started = time.perf_counter()
                    receipt = await asyncio.shield(asyncio.wrap_future(pending[0]))
                    pending.pop(0)
                    yield receipt
                    self.backpressure_seconds += time.perf_counter() - started
                slot = self.pool.acquire()
                if slot is None:
                    raise RuntimeError("Capture admission violated the two-slot bound")
                try:
                    # Shield CUDA submissions: cancellation must not release a live slot.
                    capture = loop.run_in_executor(self.producer, self._capture, tokens, slot)
                    try:
                        tensors, copied, events = await asyncio.shield(capture)
                    except asyncio.CancelledError:
                        tensors, copied, events = await capture
                        await asyncio.to_thread(self._finish, copied, events)
                        raise
                    pending.append(self.writer.submit(self._publish, tokens, slot, tensors, copied, events,
                                                       Path(output_dir) / f"{index:06d}.safetensors", index))
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
                results = await asyncio.shield(asyncio.gather(
                    *(asyncio.wrap_future(future) for future in pending), return_exceptions=True))
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

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
