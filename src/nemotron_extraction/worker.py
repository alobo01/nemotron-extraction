"""Nemotron hooks inside a vLLM worker; RPCs never wait for shard writes."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import torch

from .hooks import install, layers_of
from .precision import fp32_router, ieee_settings, precision_settings, set_precision
from .transport import (
    SLOTS,
    BufferPool,
    capture_seconds,
    device_finite,
    finish_copy,
    memory_peaks,
    publish,
    publish_site_directory,
    seconds,
    timeline,
)
from .types import HookSite, validate_publication_threads


def external_request_id(rid, known):
    """Map a scheduler request ID back to the caller's ID."""
    if rid in known:
        return rid
    # vLLM 0.20.1 through 0.27.1 append an eight-character internal uniqueness suffix.
    external, separator, suffix = rid.rpartition("-")
    if not separator or len(suffix) != 8 or external not in known:
        raise RuntimeError(f"Unknown request ID {rid}")
    return external


class ActivationHookExtension:
    def nh_install(
        self,
        sites,
        sink_dir,
        diagnostics=False,
        publication_threads=1,
        publication_layout="monolithic",
        site_write_threads=8,
        capture_slots=SLOTS,
    ):
        validate_publication_threads(publication_threads)
        if publication_layout not in ("monolithic", "per_site") or not 1 <= site_write_threads <= 8:
            raise ValueError("Invalid publication layout or site writer count")
        self.nh_publication_layout = publication_layout
        self.nh_site_writer = ThreadPoolExecutor(max_workers=site_write_threads)
        self.nh_diagnostics = diagnostics
        self.nh_captured = {}
        self.nh_capture_events = []
        self.nh_pool = BufferPool(capture_slots)
        # Open capture waves and the scheduler request IDs that route rows into them.
        self.nh_waves = {}
        self.nh_requests = {}
        self.nh_spans = []
        self.nh_pending = {}
        self.nh_copy = ThreadPoolExecutor(max_workers=1)
        self.nh_writer = ThreadPoolExecutor(max_workers=publication_threads)
        self.nh_stream = torch.cuda.Stream()
        self.nh_origin = torch.cuda.Event(enable_timing=True)
        self.nh_origin.record()
        torch.cuda.synchronize()
        self.nh_origin_cpu = time.perf_counter()
        specs = [HookSite(**site) for site in sites]
        self.nh_keys = [site.key for site in specs]
        self.nh_enabled_keys = list(self.nh_keys)
        model = self.model_runner.model
        try:
            self.nh_handles, self.nh_topology = install(
                model,
                specs,
                self.nh_captured,
                worker=True,
                capture_events=self.nh_capture_events if diagnostics else None,
            )
            self.nh_handles.extend(
                [
                    model.register_forward_pre_hook(self._nh_inputs, with_kwargs=True),
                    model.register_forward_hook(self._nh_end),
                ]
            )
        except BaseException:
            self.nh_copy.shutdown()
            self.nh_writer.shutdown()
            self.nh_site_writer.shutdown()
            raise
        torch.cuda.reset_peak_memory_stats()
        return self.nh_topology

    def nh_match_precision(self, checkpoint_dir):
        """IEEE float32 matmuls and a float32 router, as in the PyTorch reference."""
        set_precision(ieee_settings())
        _, layers = layers_of(self.model_runner.model)
        self.nh_router_forwards = fp32_router(layers, checkpoint_dir)
        return dict(settings=precision_settings(), fp32_router_layers=sorted(self.nh_router_forwards))

    def nh_memory_window(self, release_cached=False):
        """Reset phase peaks only after captured tensors and writers have drained."""
        if self.nh_waves or self.nh_pending:
            raise RuntimeError("Drain publications before resetting memory statistics")
        torch.cuda.synchronize()
        if release_cached:
            torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        snapshot = memory_peaks()
        snapshot["external_device_memory_bytes"] = max(
            0, snapshot["peak_device_memory_bytes"] - torch.cuda.memory_reserved()
        )
        return snapshot

    def nh_set_sites(self, keys):
        if self.nh_waves or self.nh_pending:
            raise RuntimeError("Drain publications before changing active sites")
        if not keys or len(set(keys)) != len(keys) or not set(keys) <= set(self.nh_keys):
            raise ValueError("Active sites must be a nonempty subset of installed hooks")
        self.nh_enabled_keys = list(keys)
        return True

    def nh_begin(self, wave_id, request_ids, token_ids, path):
        if wave_id in self.nh_waves or wave_id in self.nh_pending:
            raise RuntimeError("Capture wave already active")
        if any(rid in self.nh_requests for rid in request_ids):
            raise ValueError("Request ID already belongs to an open capture wave")
        slot = self.nh_pool.acquire()
        if slot is None:
            return False
        tokens = torch.tensor(token_ids, dtype=torch.long)
        if len(request_ids) != len(tokens) or len(set(request_ids)) != len(request_ids):
            self.nh_pool.release(slot)
            raise ValueError("Each prompt requires a unique request ID")
        wave = {
            "id": wave_id,
            "slot": slot,
            "tokens": tokens,
            "requests": {rid: i for i, rid in enumerate(request_ids)},
            "path": path,
            "tensors": {"token_ids": tokens},
            "copies": [],
            "finite_flags": [],
            "layout": self.nh_publication_layout,
            "keys": list(self.nh_enabled_keys),
            "site_writer": self.nh_site_writer,
            # Next prompt offset per row: spans must arrive contiguous and in order.
            "cursor": [0] * len(tokens),
            "forward_events": [],
            "forward_batches": [],
            "copy_events": [],
            "origin": self.nh_origin,
            "scheduler": [],
            "diagnostics": self.nh_diagnostics,
            "capture_events": [],
            "origin_cpu": self.nh_origin_cpu,
            "memory": {},
        }
        self.nh_waves[wave_id] = wave
        self.nh_requests.update((rid, (wave, row)) for rid, row in wave["requests"].items())
        return True

    def _nh_inputs(self, module, args, kwargs):
        if not self.nh_waves:
            raise RuntimeError("Inference submitted without a capture reservation")
        runner = self.model_runner
        batch = runner.input_batch
        ids = list(batch.req_ids)
        # CPU scheduler metadata: no CUDA synchronization or token-content matching.
        starts = runner.query_start_loc.cpu[: len(ids) + 1].tolist()
        spans = []
        for index, rid in enumerate(ids):
            rid = external_request_id(rid, self.nh_requests)
            wave, row = self.nh_requests[rid]
            computed = int(batch.num_computed_tokens_cpu[index])
            length = wave["tokens"].shape[1]
            start, end = int(starts[index]), int(starts[index + 1])
            if computed >= length:
                continue  # generated tokens are never captured
            if computed != wave["cursor"][row]:
                expected = wave["cursor"][row]
                raise RuntimeError(f"Prompt span out of order for {rid}: expected offset {expected}, got {computed}")
            count = min(end - start, length - computed)
            if self.nh_diagnostics:
                token_ids = batch.token_ids_cpu[index, computed : computed + count]
                if token_ids.tolist() != wave["tokens"][row, computed : computed + count].tolist():
                    raise RuntimeError("Worker input token IDs differ from submitted prompt")
            spans.append((wave, row, computed, start, count, rid))
        # Commit only after every request in this forward was accepted.
        for wave, row, computed, start, count, rid in spans:
            wave["cursor"][row] = computed + count
            wave["scheduler"].append(
                dict(
                    request_id=rid,
                    sequence_index=row,
                    start=start,
                    stop=start + count,
                    computed_tokens=computed,
                    prompt_length=wave["tokens"].shape[1],
                )
            )
        self.nh_spans = [span[:5] for span in spans]
        self.nh_forward_start = torch.cuda.Event(enable_timing=True)
        self.nh_forward_start.record()

    def _nh_end(self, module, args, output):
        end_forward = torch.cuda.Event(enable_timing=True)
        end_forward.record()
        captured = dict(self.nh_captured)
        self.nh_captured.clear()
        capture_events = list(self.nh_capture_events)
        self.nh_capture_events.clear()
        if set(captured) != set(self.nh_keys):
            raise RuntimeError("Captured site set differs from requested sites")
        if not self.nh_spans:
            return
        captured = {key: captured[key] for key in self.nh_enabled_keys}
        # Waves sharing one forward split its forward and copy time in proportion to their rows.
        rows = {}
        for wave, _, _, _, count in self.nh_spans:
            rows[wave["id"]] = rows.get(wave["id"], 0) + count
        total = sum(rows.values())
        waves = {wave["id"]: wave for wave, *_ in self.nh_spans}
        copy, events = self._nh_transfer(captured, waves)
        for wave_id, wave in waves.items():
            share = rows[wave_id] / total
            wave["forward_events"].append((self.nh_forward_start, end_forward, share))
            wave["forward_batches"].append(sum(1 for span in self.nh_spans if span[0] is wave))
            wave["copies"].append((copy, share))
            wave["copy_events"].append(events)
            wave["capture_events"].extend(capture_events)
            if self.nh_diagnostics:
                wave["memory"] = memory_peaks(wave["memory"])
        self.nh_spans = []

    def _nh_transfer(self, captured, waves):
        """Copy this forward's spans into each wave's pinned buffers; return the copy future and events."""
        last = max(start + count for _, _, _, start, count in self.nh_spans)
        stream = torch.cuda.current_stream()
        started, ended = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        with torch.cuda.stream(self.nh_stream):
            self.nh_stream.wait_stream(stream)
            started.record()
            for key, source in captured.items():
                if source.ndim != 2 or last > source.shape[0]:
                    raise RuntimeError(f"{key}: capture shape does not cover scheduler spans")
                self.nh_topology[key]["width"] = source.shape[-1]
                finite = device_finite(source[:last]) if self.nh_publication_layout == "per_site" else None
                for wave in waves.values():
                    if finite is not None:
                        flag = self.nh_pool.tensor(
                            wave["slot"], key + "_finite_" + str(len(wave["finite_flags"])), (), torch.bool, pinned=True
                        )
                        flag.copy_(finite, non_blocking=True)
                        # A wave may contain several forwards; keep each result independently.
                        wave["finite_flags"].append((key, flag))
                    wave["tensors"][key] = self.nh_pool.tensor(
                        wave["slot"], key, (wave["tokens"].numel(), source.shape[-1]), source.dtype, pinned=True
                    )
                for wave, row, offset, start, count in self.nh_spans:
                    destination = row * wave["tokens"].shape[1] + offset
                    wave["tensors"][key][destination : destination + count].copy_(
                        source[start : start + count], non_blocking=True
                    )
                source.record_stream(self.nh_stream)
            ended.record()
        return self.nh_copy.submit(finish_copy, started, ended, captured), (started, ended)

    def nh_seal(self, wave_id):
        wave = self.nh_waves.get(wave_id)
        if wave is None:
            raise RuntimeError("Unknown capture wave")
        length = wave["tokens"].shape[1]
        if any(position != length for position in wave["cursor"]):
            raise RuntimeError("Missing prompt activations")
        del self.nh_waves[wave_id]
        for rid in wave["requests"]:
            del self.nh_requests[rid]
        wave["memory"] = memory_peaks(wave["memory"])
        wave["topology"] = {key: dict(value) for key, value in self.nh_topology.items()}
        self.nh_pending[wave_id] = (wave["slot"], self.nh_writer.submit(self._nh_publish, wave))
        return True

    @staticmethod
    def _nh_publish(wave):
        copy_seconds = sum(copy.result() * share for copy, share in wave["copies"])
        forward_seconds = sum(seconds(start, end) * share for start, end, share in wave["forward_events"])
        write_start = time.perf_counter() - wave["origin_cpu"]
        if wave.get("layout") == "per_site":
            for key, flag in wave["finite_flags"]:
                if not bool(flag):
                    raise ValueError(f"{key}: nonfinite activations")
            receipt = publish_site_directory(
                wave["tensors"], wave["path"], wave["site_writer"], {"wave_id": wave["id"]}
            )
        else:
            receipt = publish(wave["tensors"], wave["path"], {"wave_id": wave["id"]})
        receipt.update(
            wave["memory"],
            wave_id=wave["id"],
            tokens=wave["tokens"].numel(),
            capture_seconds=capture_seconds(wave["capture_events"]) if wave["diagnostics"] else None,
            worker_copy_seconds=copy_seconds,
            worker_forward_seconds=forward_seconds,
            actual_forward_batches=wave["forward_batches"],
            topology=wave["topology"],
            scheduler=wave["scheduler"],
            diagnostics=wave["diagnostics"],
        )
        receipt["timeline"] = timeline(
            wave["origin"],
            (("forward", [(start, end) for start, end, _ in wave["forward_events"]]), ("D2H", wave["copy_events"])),
        )
        receipt["timeline"].append(
            {"name": "write", "start": write_start, "end": time.perf_counter() - wave["origin_cpu"]}
        )
        return receipt

    def nh_poll(self):
        """Return only completed publications; never wait on a future here."""
        ready = []
        for wave_id, (slot, future) in list(self.nh_pending.items()):
            if future.done():
                del self.nh_pending[wave_id]
                try:
                    ready.append(future.result())
                finally:
                    self.nh_pool.release(slot)
        return ready

    def nh_remove(self):
        try:
            self.nh_copy.shutdown(wait=True)
            self.nh_writer.shutdown(wait=True)
            self.nh_site_writer.shutdown(wait=True)
            for _, future in self.nh_pending.values():
                future.result()
        finally:
            for handle in self.nh_handles:
                handle.remove()
            if getattr(self, "nh_router_forwards", None):
                _, layers = layers_of(self.model_runner.model)
                for number, forward in self.nh_router_forwards.items():
                    layers[number].mixer.gate.forward = forward
            self.nh_handles.clear()
            self.nh_captured.clear()
            self.nh_waves.clear()
            self.nh_requests.clear()
            self.nh_pending.clear()
            self.nh_pool.reset()
