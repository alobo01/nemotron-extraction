"""Unchanged nemotron-sae extraction methods, driven for comparison with this package.

A snapshot directory holds copies of the sibling's worker.py and attention.py.
Only their definitions execute; training, queue and remote-communication imports
are excluded, and capture, row assembly and persistence run unmodified.
"""
from __future__ import annotations

import ast
import asyncio
from collections.abc import Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from functools import partial
import json
import os
from pathlib import Path
import time

import torch
from torch import nn
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from nemotron_extraction import HookSite
from nemotron_extraction.hooks import resolve

from .instrumentation import overlap_seconds


def configured_kv_cache_bytes():
    value = os.environ.get("NH_KV_CACHE_BYTES")
    return int(value) if value else None


def configured_chunked_prefill():
    value = os.environ.get("NH_ENABLE_CHUNKED_PREFILL", "1").lower()
    if value not in ("1", "true", "yes", "0", "false", "no"):
        raise ValueError("NH_ENABLE_CHUNKED_PREFILL must be a boolean")
    return value in ("1", "true", "yes")


def sibling_namespace(directory):
    """Execute the snapshot's definitions only; its training and queue imports never run."""
    scope = dict(torch=torch, nn=nn, partial=partial, Path=Path, json=json, time=time,
                 ThreadPoolExecutor=ThreadPoolExecutor, Future=Future, Iterable=Iterable, save_file=save_file)
    for filename in ('attention.py', 'worker.py'):
        path = Path(directory)/filename
        tree = ast.parse(path.read_text())
        nodes = [ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)]
        nodes += [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) or
                  isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in
                  ('MIXER_KINDS','RESIDUAL_KINDS') for t in n.targets)]
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(path), 'exec'), scope)
    required = ("sae_install_hooks", "_sae_end_step", "sae_save", "sae_flush_saves", "sae_flush")
    if not all(hasattr(scope["ActivationHookExtension"], name) for name in required):
        raise ValueError("Sibling source lacks the expected file/fill interface")
    return scope


class SiblingWorker:
    """Instrumentation delegates extraction to the snapshot, without rewriting it."""
    def cmp_install(self, reference, sites, sink, mode):
        source = sibling_namespace(reference)
        self.cmp_source = source
        self.cmp_worker = source["ActivationHookExtension"]()
        self.cmp_worker.model_runner = self.model_runner
        self.cmp_events, self.cmp_batches, self.cmp_scheduler = [], [], []
        model = self.model_runner.model
        self.cmp_handles = [model.register_forward_pre_hook(self._cmp_start, with_kwargs=True)]
        specs = [(s["key"], s["kind"], s["layer"]) for s in sites]
        self.cmp_worker.sae_install_hooks(specs, sink if mode == "sibling_files" else None)
        # Stop the GPU forward timer before the sibling's CPU copies and writes.
        self.cmp_handles.append(model.register_forward_hook(self._cmp_stop, prepend=True))
        topology = {}
        for spec in sites:
            site = HookSite(spec["layer"], spec["kind"])
            path, _ = resolve(model, site)
            topology[site.key] = dict(layer=site.layer, kind=site.kind, module_path=path)
        return topology

    def _cmp_start(self, module, args, kwargs):
        self.cmp_start = torch.cuda.Event(enable_timing=True)
        self.cmp_start.record()
        # Read scheduler metadata on the CPU; do not synchronize for instrumentation.
        runner = self.model_runner
        counts = runner.input_batch.num_computed_tokens_cpu[:len(runner.input_batch.req_ids)]
        self.cmp_batches.append(sum(int(n) == 0 for n in counts))
        starts = runner.query_start_loc.cpu[:len(counts)+1].tolist()
        for i, rid in enumerate(runner.input_batch.req_ids):
            computed = int(counts[i])
            if computed >= self.cmp_prompt_length:
                continue
            if computed != 0 or starts[i+1]-starts[i] != self.cmp_prompt_length:
                raise RuntimeError('Sibling comparison requires whole-prefill prompts')
            self.cmp_scheduler.append(dict(request_id=rid, start=int(starts[i]), stop=int(starts[i+1]),
                computed_tokens=computed, prompt_length=self.cmp_prompt_length))

    def cmp_expect(self, prompt_length):
        self.cmp_prompt_length = prompt_length

    def _cmp_stop(self, module, args, output):
        stop = torch.cuda.Event(enable_timing=True)
        stop.record()
        self.cmp_events.append((self.cmp_start, stop))

    def cmp_files(self):
        return self.cmp_worker.sae_flush()

    def cmp_save(self, tokens, path):
        return self.cmp_worker.sae_save(tokens, path, 4)

    def cmp_drain(self):
        return self.cmp_worker.sae_flush_saves()

    def cmp_stats(self, reset=False):
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        result = dict(forward_seconds=sum(a.elapsed_time(b) / 1000 for a, b in self.cmp_events),
                      actual_forward_batches=list(self.cmp_batches),
                      scheduler=list(self.cmp_scheduler),
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(), device_used_bytes=total-free)
        if reset:
            self.cmp_events.clear()
            self.cmp_batches.clear()
            self.cmp_scheduler.clear()
            torch.cuda.reset_peak_memory_stats()
        return result

    def cmp_close(self):
        try:
            self.cmp_worker.sae_remove_hooks()
        finally:
            for handle in self.cmp_handles:
                handle.remove()


class SiblingExtractor:
    """Original generate/flush/read and generate/save/drain driver protocols."""
    def __init__(self, model, sites, reference, mode, batch_size, sequence_length, sink):
        from vllm import LLM, SamplingParams
        self.mode, self.sites, self.reference = mode, sites, reference
        self.sink = Path(sink)
        self.sink.mkdir(parents=True, exist_ok=True)
        self.generate_seconds = self.copy_seconds = self.write_seconds = 0.
        self.driver_read_seconds = self.final_drain_seconds = 0.
        self.assemble_rows = sibling_namespace(reference)["assemble_rows"]
        self.installed = False
        started = time.perf_counter()
        self.llm = LLM(model=model, dtype="bfloat16", trust_remote_code=True,
            tensor_parallel_size=1, enforce_eager=True, enable_prefix_caching=False,
            enable_chunked_prefill=configured_chunked_prefill(), skip_tokenizer_init=True,
            max_model_len=max(2049, sequence_length + 1), max_num_seqs=batch_size,
            max_num_batched_tokens=batch_size * max(2049, sequence_length + 1),
            gpu_memory_utilization=.96, kv_cache_memory_bytes=configured_kv_cache_bytes(),
            worker_extension_cls="benchlib.sibling.SiblingWorker")
        self.engine_startup_seconds = time.perf_counter() - started
        started = time.perf_counter()
        self.topology = self.rpc("cmp_install", str(reference),
            [dict(key=s.key, layer=s.layer, kind=s.kind) for s in sites], str(self.sink), mode)
        self.installed = True
        self.hook_install_seconds = time.perf_counter() - started
        self.params = SamplingParams(max_tokens=1, temperature=0)

    def rpc(self, method, *args):
        return self.llm.collective_rpc(method, args=args)[0]

    def saved(self, receipt):
        if receipt:
            self.copy_seconds += receipt["copy_seconds"]
            self.write_seconds += receipt["write_seconds"]

    async def stream_to_disk(self, token_batches, output_dir):
        receipts = []
        for index, tokens in enumerate(token_batches):
            self.rpc('cmp_expect', tokens.shape[1])
            started = time.perf_counter()
            self.llm.generate([{"prompt_token_ids": row.tolist()} for row in tokens],
                              self.params, use_tqdm=False)
            self.generate_seconds += time.perf_counter() - started
            if self.mode == "sibling_files":
                paths = self.rpc("cmp_files")
                # Time file reads and row assembly, as the sibling's own extract() does.
                started = time.perf_counter()
                values = self.assemble_rows(
                    [load_file(p) for p in paths], tokens, [s.key for s in self.sites])
                self.driver_read_seconds += time.perf_counter() - started
                del values
                receipts.append(dict(index=index, mode=self.mode, paths=paths, tokens=tokens.numel()))
            else:
                path = Path(output_dir) / f"{index:06d}"
                self.saved(self.rpc("cmp_save", tokens.tolist(), str(path)))
                receipts.append(dict(index=index, mode=self.mode, path=str(path), tokens=tokens.numel()))
        started = time.perf_counter()
        if self.mode == "sibling_fills":
            self.saved(self.rpc("cmp_drain"))
        self.final_drain_seconds += time.perf_counter() - started
        for receipt in receipts:
            yield receipt

    def close(self):
        if self.llm is not None:
            try:
                if self.installed:
                    self.rpc("cmp_close")
            finally:
                shutdown = getattr(self.llm.llm_engine, "shutdown", None)
                if shutdown:
                    shutdown()
                self.llm = None


async def persist(extractor, token_batches, output):
    return [receipt async for receipt in extractor.stream_to_disk(token_batches, output)]


def measure(extractor, token_batches, output):
    """Stream batches to disk and report throughput, stage timings and forward/transfer overlap."""
    before = {name: getattr(extractor, name, 0.) for name in (
        "model_forward_seconds", "worker_forward_seconds", "cpu_copy_seconds", "worker_copy_seconds",
        "worker_write_seconds", "generate_seconds", "driver_read_seconds", "backpressure_seconds",
        "worker_rpc_seconds",
        "final_drain_seconds", "copy_seconds", "write_seconds")}
    if isinstance(extractor, SiblingExtractor):
        extractor.rpc("cmp_stats", True)
    actual_start = len(getattr(extractor, "actual_forward_batches", []))
    started = time.perf_counter()
    receipts = asyncio.run(persist(extractor, token_batches, output))
    elapsed = time.perf_counter() - started
    measured = {name: getattr(extractor, name, 0.) - value for name, value in before.items()}
    timeline = [event for receipt in receipts for event in receipt.get("timeline", [])]
    if isinstance(extractor, SiblingExtractor):
        measured.update(extractor.rpc("cmp_stats"))
    else:
        measured["forward_seconds"] = measured["model_forward_seconds"] + measured["worker_forward_seconds"]
        measured["actual_forward_batches"] = extractor.actual_forward_batches[actual_start:]
    count = sum(tokens.numel() for tokens in token_batches)
    measured.update(tokens=count, wall_seconds=elapsed, completed_to_disk_tokens_per_second=count/elapsed,
        forward_tokens_per_second=count/measured["forward_seconds"], receipts=receipts,
        D2H_forward_overlap_seconds=overlap_seconds(timeline, "D2H"),
        write_forward_overlap_seconds=overlap_seconds(timeline, "write"),
        overlap_instrumented=not isinstance(extractor, SiblingExtractor), timeline=timeline)
    if sum(measured["actual_forward_batches"]) != sum(len(t) for t in token_batches):
        raise ValueError("Actual prompt forward-batch counts differ from submitted sequences")
    return measured


def read_site(receipt,tokens,key,snapshot):
    if receipt.get('mode') == 'sibling_files':
        steps=[]
        for path in receipt['paths']:
            with safe_open(path,framework='pt',device='cpu') as f:
                steps.append({k:f.get_tensor(k) for k in ('input_ids','positions',key)})
        return sibling_namespace(snapshot)['assemble_rows'](steps,tokens,[key])[key]
    if receipt.get('mode') == 'sibling_fills':
        root=Path(receipt['path'])
        with safe_open(root/'token_ids.safetensors',framework='pt',device='cpu') as f:
            if not torch.equal(f.get_tensor('token_ids'),tokens): raise ValueError('Token attribution differs')
        with safe_open(root/f'{key}.safetensors',framework='pt',device='cpu') as f:
            return f.get_tensor('activations')
    with safe_open(receipt['path'],framework='pt',device='cpu') as f:
        if not torch.equal(f.get_tensor('token_ids'),tokens): raise ValueError('Token attribution differs')
        return f.get_tensor(key)
