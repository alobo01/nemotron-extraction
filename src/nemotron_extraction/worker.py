"""Nemotron hooks inside a vLLM worker; RPCs never wait for shard writes."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
import time
import torch
from .hooks import install
from .transport import BufferPool, capture_seconds, finish_copy, memory_peaks, publish, seconds, timeline
from .types import HookSite, validate_publication_threads


def external_request_id(rid, known):
    """Map a scheduler request ID back to the caller's ID."""
    if rid in known:
        return rid
    # vLLM 0.20.1 appends an eight-character internal uniqueness suffix.
    external, separator, suffix = rid.rpartition("-")
    if not separator or len(suffix) != 8 or external not in known:
        raise RuntimeError(f"Unknown request ID {rid}")
    return external


class ActivationHookExtension:
    def nh_install(self, sites, sink_dir, diagnostics=False, publication_threads=1):
        validate_publication_threads(publication_threads)
        self.nh_diagnostics = diagnostics
        self.nh_captured = {}
        self.nh_capture_events = []
        self.nh_pool = BufferPool()
        self.nh_active = None
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
        model = self.model_runner.model
        try:
            self.nh_handles, self.nh_topology = install(model, specs, self.nh_captured, worker=True,
                capture_events=self.nh_capture_events if diagnostics else None)
            self.nh_handles.extend([
                model.register_forward_pre_hook(self._nh_inputs, with_kwargs=True),
                model.register_forward_hook(self._nh_end),
            ])
        except BaseException:
            self.nh_copy.shutdown()
            self.nh_writer.shutdown()
            raise
        torch.cuda.reset_peak_memory_stats()
        return self.nh_topology

    def nh_begin(self, wave_id, request_ids, token_ids, path):
        if self.nh_active is not None or wave_id in self.nh_pending:
            raise RuntimeError("Capture wave already active")
        slot = self.nh_pool.acquire()
        if slot is None:
            return False
        tokens = torch.tensor(token_ids, dtype=torch.long)
        if len(request_ids) != len(tokens) or len(set(request_ids)) != len(request_ids):
            self.nh_pool.release(slot)
            raise ValueError("Each prompt requires a unique request ID")
        self.nh_active = {"id": wave_id, "slot": slot, "tokens": tokens,
                          "requests": {rid: i for i, rid in enumerate(request_ids)},
                          "path": path, "tensors": {"token_ids": tokens}, "copies": [],
                          "covered": set(), "forward_events": [], "forward_batches": [],
                          "copy_events": [], "origin": self.nh_origin,
                          "scheduler": [], "diagnostics": self.nh_diagnostics,
                          "capture_events": [], "origin_cpu": self.nh_origin_cpu, "memory": {}}
        return True

    def _nh_inputs(self, module, args, kwargs):
        if self.nh_active is None:
            raise RuntimeError("Inference submitted without a capture reservation")
        wave = self.nh_active
        runner = self.model_runner
        batch = runner.input_batch
        ids = list(batch.req_ids)
        # CPU scheduler metadata: no CUDA synchronization or token-content matching.
        starts = runner.query_start_loc.cpu[:len(ids) + 1].tolist()
        spans = []
        length = wave["tokens"].shape[1]
        for index, rid in enumerate(ids):
            computed = int(batch.num_computed_tokens_cpu[index])
            rid = external_request_id(rid, wave["requests"])
            prompt_index = wave["requests"][rid]
            start, end = int(starts[index]), int(starts[index + 1])
            if computed >= length:
                continue
            if computed != 0 or end - start != length:
                raise RuntimeError("Requires whole-prefill prompts; partial prefill is unsupported")
            if prompt_index in wave["covered"]:
                raise RuntimeError("Prompt captured twice")
            if self.nh_diagnostics and not torch.equal(
                    batch.token_ids_cpu[index, :length], wave["tokens"][prompt_index]):
                raise RuntimeError("Worker input token IDs differ from submitted prompt")
            spans.append((prompt_index, start, end))
            wave["scheduler"].append(dict(request_id=rid, sequence_index=prompt_index,
                start=start, stop=end, computed_tokens=computed, prompt_length=length))
            wave["covered"].add(prompt_index)
        self.nh_spans = spans
        self.nh_forward_start = torch.cuda.Event(enable_timing=True)
        self.nh_forward_start.record()

    def _nh_end(self, module, args, output):
        wave = self.nh_active
        end_forward = torch.cuda.Event(enable_timing=True)
        end_forward.record()
        wave["forward_events"].append((self.nh_forward_start, end_forward))
        wave["forward_batches"].append(len(self.nh_spans))
        captured = dict(self.nh_captured)
        self.nh_captured.clear()
        wave["capture_events"].extend(self.nh_capture_events)
        self.nh_capture_events.clear()
        if set(captured) != set(self.nh_keys):
            raise RuntimeError("Captured site set differs from requested sites")
        stream = torch.cuda.current_stream()
        started, ended = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        total_rows = wave["tokens"].numel()
        length = wave["tokens"].shape[1]
        with torch.cuda.stream(self.nh_stream):
            self.nh_stream.wait_stream(stream)
            started.record()
            for key, source in captured.items():
                if source.ndim != 2 or any(stop > source.shape[0] for _, _, stop in self.nh_spans):
                    raise RuntimeError(f"{key}: capture shape does not cover scheduler spans")
                host = self.nh_pool.tensor(wave["slot"], key, (total_rows, source.shape[-1]),
                                           source.dtype, pinned=True)
                wave["tensors"][key] = host
                self.nh_topology[key]["width"] = source.shape[-1]
                for index, start, stop in self.nh_spans:
                    host[index * length:(index + 1) * length].copy_(source[start:stop], non_blocking=True)
                source.record_stream(self.nh_stream)
            ended.record()
        wave["copies"].append(self.nh_copy.submit(finish_copy, started, ended, captured))
        wave["copy_events"].append((started, ended))
        if self.nh_diagnostics:
            wave["memory"] = memory_peaks(wave["memory"])

    def nh_seal(self, wave_id):
        wave = self.nh_active
        if wave is None or wave["id"] != wave_id:
            raise RuntimeError("Unknown capture wave")
        if len(wave["covered"]) != len(wave["tokens"]):
            raise RuntimeError("Missing prompt activations")
        wave["memory"] = memory_peaks(wave["memory"])
        wave["topology"] = {key: dict(value) for key, value in self.nh_topology.items()}
        self.nh_pending[wave_id] = (wave["slot"], self.nh_writer.submit(self._nh_publish, wave))
        self.nh_active = None
        return True

    @staticmethod
    def _nh_publish(wave):
        copy_seconds = sum(copy.result() for copy in wave["copies"])
        forward_seconds = sum(seconds(start, end) for start, end in wave["forward_events"])
        write_start = time.perf_counter() - wave["origin_cpu"]
        receipt = publish(wave["tensors"], wave["path"], {"wave_id": wave["id"]})
        receipt.update(wave["memory"], wave_id=wave["id"], tokens=wave["tokens"].numel(),
                       capture_seconds=capture_seconds(wave["capture_events"]) if wave["diagnostics"] else None,
                       worker_copy_seconds=copy_seconds, worker_forward_seconds=forward_seconds,
                       actual_forward_batches=wave["forward_batches"], topology=wave["topology"],
                       scheduler=wave["scheduler"], diagnostics=wave["diagnostics"])
        receipt["timeline"] = timeline(wave["origin"], (("forward", wave["forward_events"]),
                                                        ("D2H", wave["copy_events"])))
        receipt["timeline"].append({"name": "write", "start": write_start,
                                    "end": time.perf_counter() - wave["origin_cpu"]})
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
            for _, future in self.nh_pending.values():
                future.result()
        finally:
            for handle in self.nh_handles:
                handle.remove()
            self.nh_handles.clear()
            self.nh_captured.clear()
            self.nh_active = None
            self.nh_pending.clear()
            self.nh_pool.reset()
