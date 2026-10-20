# /// script
# requires-python = ">=3.10"
# dependencies = ["nemotron-extraction[hf,vllm,benchmark]==1.1.0"]
# ///
"""Compare installed hooks with unchanged sibling file/fill extraction methods.

The reference directory contains snapshots of vllm_worker.py and attention.py
from nemotron-sae. Only their definitions execute: training, queues and remote
communication imports are excluded. Capture, row assembly and persistence
methods are unchanged. Infrastructure settings do not belong in this script.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import traceback

import torch
from safetensors.torch import load_file, save_file
from nemotron_extraction import HFExtractor, HookSite, VLLMExtractor
from benchlib.sibling import sibling_namespace as load_sibling
from nemotron_extraction.hooks import representative_sites, resolve
from targeted_validation import Utilisation, overlap_seconds
from validate import Agreement, MODEL, batches, sha256, versions, write_json

METHODS = ("transformers", "package_async", "sibling_files", "sibling_fills")


def configured_kv_cache_bytes():
    value = os.environ.get("NH_KV_CACHE_BYTES")
    return int(value) if value else None


def configured_chunked_prefill():
    value = os.environ.get("NH_ENABLE_CHUNKED_PREFILL", "1").lower()
    if value not in ("1", "true", "yes", "0", "false", "no"):
        raise ValueError("NH_ENABLE_CHUNKED_PREFILL must be a boolean")
    return value in ("1", "true", "yes")


def sibling_namespace(directory):
    """Retain original function/class bodies; exclude unrelated package imports."""
    scope = load_sibling(directory)
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
            worker_extension_cls="compare_repositories.SiblingWorker")
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
                # Match the sibling extract() API: include file reads and row assembly.
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


def cases(tokens, batch_size, sequence_length, count):
    if count < batch_size * sequence_length or count % (batch_size * sequence_length):
        raise ValueError("Timing workload must contain full requested batches")
    if count > tokens.numel():
        raise ValueError("Insufficient frozen tokens")
    return list(batches(tokens[:count], batch_size, sequence_length))


def load_capture(receipt, tokens, sites, reference):
    if receipt.get("mode") == "sibling_files":
        return sibling_namespace(reference)["assemble_rows"](
            [load_file(p) for p in receipt["paths"]], tokens, [s.key for s in sites])
    if receipt.get("mode") == "sibling_fills":
        directory = Path(receipt["path"])
        if not torch.equal(load_file(str(directory / "token_ids.safetensors"))["token_ids"], tokens):
            raise ValueError("Sibling fill token order differs")
        return {s.key: load_file(str(directory / f"{s.key}.safetensors"))["activations"] for s in sites}
    values = load_file(receipt["path"])
    if not torch.equal(values.pop("token_ids"), tokens):
        raise ValueError("Persisted token order differs")
    return values


def compare_receipts(reference_receipts, candidate_receipts, token_batches, sites, snapshot, diagnostic):
    if len(reference_receipts) != len(candidate_receipts) or len(token_batches) != len(candidate_receipts):
        raise ValueError("Reference and candidate batch counts differ")
    metrics = {s.key: Agreement() for s in sites}
    sample = {}
    for index, (left_receipt, right_receipt, tokens) in enumerate(zip(reference_receipts, candidate_receipts, token_batches)):
        left = load_capture(left_receipt, tokens, sites, snapshot)
        right = load_capture(right_receipt, tokens, sites, snapshot)
        if left.keys() != right.keys():
            raise ValueError("Captured site sets differ")
        for key in metrics:
            if left[key].shape[0] != tokens.numel():
                raise ValueError("Capture row count differs from prompt token count")
            metrics[key].add(left[key], right[key])
            if index == 0:
                sample[key + "_reference"] = left[key][:8].clone()
                sample[key + "_candidate"] = right[key][:8].clone()
    save_file(sample, str(diagnostic))
    results = {key: metric.result() for key, metric in metrics.items()}
    # Population standard deviation is across token rows, not across sites.
    for key, metric in metrics.items():
        results[key]["cosine_std"] = float(torch.cat(metric.cosines).std(correction=0))
    return results


async def persist(extractor, token_batches, output):
    return [receipt async for receipt in extractor.stream_to_disk(token_batches, output)]


def phase(extractor, token_batches, output):
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


def run(args):
    torch.set_num_threads(4)
    torch.manual_seed(0)
    destination = args.output / f"b{args.batch_size}" / args.method
    destination.mkdir(parents=True, exist_ok=True)
    receipt_path = destination / "result.json"
    if receipt_path.exists():
        raise FileExistsError("Preserve existing results; use a new output directory")
    manifest = json.loads((args.work / "manifest.json").read_text())
    if manifest["model"] != MODEL or manifest["dtype"] != "bfloat16":
        raise ValueError("Requires the frozen official BF16 checkpoint")
    token_sets = {}
    for language in ("en", "de", "es"):
        path = args.work / f"{language}.safetensors"
        if sha256(path) != manifest["languages"][language]["token_sha256"]:
            raise ValueError("Frozen token checksum differs")
        token_sets[language] = load_file(str(path))["token_ids"]
    timing_batches = cases(token_sets["en"], args.batch_size, args.sequence_length, args.tokens)
    device = torch.cuda.get_device_properties(0)
    record = dict(status="running", method=args.method, batch_size=args.batch_size, software=versions(),
        gpu=dict(name=device.name, total_memory_bytes=device.total_memory, uuid=str(device.uuid)),
        sequence_length=args.sequence_length, timing_language="en", model_revision=manifest["model_revision"],
        tokenizer_revision=manifest["tokenizer_revision"], manifest_sha256=sha256(args.work / "manifest.json"),
        token_hashes={language: data["token_sha256"] for language, data in manifest["languages"].items()},
        reference_source_sha256={p.name:sha256(p) for p in sorted(args.reference.glob("*.py"))},
        script_sha256=sha256(__file__), memory_target_fraction=.96, seed=0, controls={}, measurements={},
        package_code_sha256={p.name:sha256(p) for p in sorted(
            Path(__import__('nemotron_extraction').__file__).parent.glob('*.py'))},
        engine_configuration=dict(enforce_eager=True, tensor_parallel_size=1, enable_prefix_caching=False,
            enable_chunked_prefill=configured_chunked_prefill(), max_model_len=max(2049,args.sequence_length+1),
            max_num_seqs=args.batch_size, max_num_batched_tokens=args.batch_size*max(2049,args.sequence_length+1),
            kv_cache_memory_bytes=configured_kv_cache_bytes())
            if args.method != 'transformers' else dict(use_cache=False))
    extractor = None
    monitor = Utilisation(destination / "utilisation.jsonl")
    current_phase = "startup"
    begun = time.perf_counter()
    try:
        with monitor:
            site_path = args.output / "sites.json"
            if args.method == "transformers":
                from transformers import AutoModelForCausalLM
                model = AutoModelForCausalLM.from_pretrained(manifest["model_path"],
                    torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True, local_files_only=True)
                sites = representative_sites(model)
                extractor = HFExtractor(model, sites)
                current_sites = [dict(layer=s.layer, kind=s.kind) for s in sites]
                if site_path.exists() and json.loads(site_path.read_text()) != current_sites:
                    raise ValueError("Resolved reference sites changed")
                write_json(site_path, current_sites)
            else:
                sites = [HookSite(**s) for s in json.loads(site_path.read_text())]
                if args.method == "package_async":
                    extractor = VLLMExtractor(manifest["model_path"], sites, sequences_per_call=args.batch_size,
                        max_model_len=max(2049, args.sequence_length+1), sink_dir=str(destination),
                        gpu_memory_utilization=.96, kv_cache_memory_bytes=configured_kv_cache_bytes(),
                        enable_chunked_prefill=configured_chunked_prefill())
                else:
                    extractor = SiblingExtractor(manifest["model_path"], sites, args.reference, args.method,
                        args.batch_size, args.sequence_length, destination / "raw")
            record["startup_seconds"] = time.perf_counter() - begun
            record["hook_install_seconds"] = extractor.hook_install_seconds
            record["engine_or_model_load_seconds"] = record["startup_seconds"] - extractor.hook_install_seconds
            current_phase = monitor.phase = "warmup"
            with tempfile.TemporaryDirectory(dir=destination) as directory:
                started = time.perf_counter()
                # Both asynchronous buffer slots and the full requested shape.
                asyncio.run(persist(extractor, [timing_batches[0], timing_batches[0]], Path(directory)))
                record["warmup_seconds"] = time.perf_counter() - started
            # File-mode raw warmup files are no longer needed.
            if isinstance(extractor, SiblingExtractor) and extractor.mode == "sibling_files":
                for path in extractor.sink.glob("step_*.safetensors"):
                    path.unlink()
            current_phase = monitor.phase = "timing"
            measurement = phase(extractor, timing_batches, destination / "timing")
            record["measurements"] = measurement
            write_json(receipt_path, record)
            current_phase = monitor.phase = "controls"
            for language, tokens in token_sets.items():
                rows = tokens[:8192].reshape(4, 2048)[[2, 0, 2, 1]].clone()
                control_inputs = [rows, rows[:1,:37].contiguous()]
                receipts = asyncio.run(persist(extractor, control_inputs, destination / "controls" / language))
                record["controls"][language] = receipts
            first_values = load_capture(measurement["receipts"][0], timing_batches[0], sites, args.reference)
            record["topology"] = {key:dict(value, width=first_values[key].shape[-1])
                                  for key,value in extractor.topology.items()}
            del first_values
            current_phase = monitor.phase = "cleanup"
            extractor.close()
            extractor = None
            record["status"] = "complete"
    except Exception as error:
        message = str(error)
        oom = isinstance(error, torch.OutOfMemoryError) or any(text in message.lower() for text in (
            "out of memory", "gpu memory budget", "no available memory", "kv cache blocks"))
        record.update(status="memory_failure" if oom else "error", failed_phase=current_phase,
                      error_type=type(error).__name__, error=message, traceback=traceback.format_exc())
    finally:
        if extractor is not None:
            try:
                extractor.close()
            except Exception as error:
                record["cleanup_error"] = str(error)
                if record["status"] == "complete":
                    record["status"] = "error"
        record["utilisation"] = monitor.summary()
        timed_samples = [s for s in monitor.samples if s['phase'] == 'timing']
        record['sampled_peak_memory_fraction'] = max(
            (s['memory_used_mib']/s['memory_total_mib'] for s in timed_samples), default=None)
        record['memory_target_exceeded'] = any(
            s['memory_used_mib'] > .96*s['memory_total_mib'] for s in timed_samples)
        if record["status"] == "complete" and (monitor.errors or "timing" not in record["utilisation"]["phases"]):
            record.update(status="error", error="Missing or failed GPU utilisation recording")
        record["process_wall_seconds"] = time.perf_counter() - begun
        write_json(receipt_path, record)
    print(json.dumps({key:record.get(key) for key in ("method","batch_size","status","failed_phase","error","startup_seconds","process_wall_seconds")}), flush=True)
    return 0 if record["status"] in ("complete","memory_failure") else 1


def compare(args):
    root = args.output / f"b{args.batch_size}"
    sites = [HookSite(**s) for s in json.loads((args.output / "sites.json").read_text())]
    records = {m:json.loads((root / m / "result.json").read_text()) for m in METHODS
               if (root / m / "result.json").exists()}
    result = dict(batch_size=args.batch_size, comparisons={}, statuses={m:records.get(m,{}).get("status","missing") for m in METHODS})
    baseline = records.get("transformers", {})
    for method, candidate in records.items():
        if method == "transformers" or candidate["status"] != "complete" or baseline.get("status") != "complete":
            continue
        for key in ("model_revision","tokenizer_revision","manifest_sha256","reference_source_sha256","script_sha256","sequence_length","token_hashes","package_code_sha256"):
            if candidate[key] != baseline[key]:
                raise ValueError(f"Provenance mismatch: {key}")
        # The three vLLM methods must have exactly the same inference runtime.
        vllm_record = records.get("package_async", {})
        if vllm_record.get("status") == "complete":
            for key in ("torch","cuda","vllm","transformers"):
                if candidate["software"][key] != vllm_record["software"][key]:
                    raise ValueError(f"vLLM runtime mismatch: {key}")
        started = time.perf_counter()
        tokens = load_file(str(args.work / "en.safetensors"))["token_ids"]
        inputs = cases(tokens, args.batch_size, args.sequence_length, args.tokens)
        stats = compare_receipts(baseline["measurements"]["receipts"], candidate["measurements"]["receipts"],
                                 inputs, sites, args.reference, root / method / "timing-diagnostic.safetensors")
        controls = {}
        for language in ("en","de","es"):
            tokens = load_file(str(args.work / f"{language}.safetensors"))["token_ids"]
            rows = tokens[:8192].reshape(4,2048)[[2,0,2,1]].clone()
            controls[language] = compare_receipts(baseline["controls"][language], candidate["controls"][language],
                [rows,rows[:1,:37].contiguous()], sites, args.reference, root / method / f"{language}-diagnostic.safetensors")
        candidate_time, baseline_time = candidate["measurements"], baseline["measurements"]
        result["comparisons"][method] = dict(agreement=stats, multilingual_controls=controls,
            completed_to_disk_speedup=candidate_time["completed_to_disk_tokens_per_second"]/baseline_time["completed_to_disk_tokens_per_second"],
            forward_speedup=candidate_time["forward_tokens_per_second"]/baseline_time["forward_tokens_per_second"],
            passed_sites=sum(s["passed"] for s in stats.values()), total_sites=len(stats),
            comparison_seconds=time.perf_counter()-started,
            runtime_difference={k:[baseline["software"][k],candidate["software"][k]] for k in ("torch","cuda","transformers")})
    # Compare vLLM implementations directly, isolating hooks from Transformers kernels.
    anchor = records.get("package_async", {})
    result["against_package_async"] = {}
    if anchor.get("status") == "complete":
        tokens = load_file(str(args.work / "en.safetensors"))["token_ids"]
        inputs = cases(tokens,args.batch_size,args.sequence_length,args.tokens)
        for method in ("sibling_files","sibling_fills"):
            if records.get(method,{}).get("status") == "complete":
                result["against_package_async"][method] = compare_receipts(anchor["measurements"]["receipts"],
                    records[method]["measurements"]["receipts"],inputs,sites,args.reference,
                    root/method/"package-diagnostic.safetensors")
                for metric in result["against_package_async"][method].values():
                    metric['mse_normalization'] = 'shared_package_vllm_rms_per_site'
    write_json(root / "comparison.json", result)
    if args.prune:
        # Only this run's explicitly selected large capture directories are removed.
        # Keep metrics, utilisation, diagnostic samples and source snapshots.
        run_root = root.resolve()
        for method in METHODS:
            for name in ("raw","timing","controls"):
                path = (root / method / name).resolve()
                if path.exists():
                    if run_root not in path.parents or path.is_symlink():
                        raise ValueError("Unsafe capture cleanup path")
                    shutil.rmtree(path)
        result["large_captures_pruned"] = True
        write_json(root / "comparison.json", result)
    report(args)


def report(args):
    rows = ["# Hook comparison", "", "All paths use the official BF16 weights and identical frozen token IDs. "
        "Timing uses English at 512 tokens per sequence; multilingual controls use 2,048 tokens and a short tail. "
        "Forward rates exclude transfers, writing and startup. Completed-to-disk rates include the final write drain. "
        "Sibling file rates also include its original driver read/assembly. Filesystem publication does not imply fsync. "
        "Startup and warmup are measured separately. Runtime differences against Transformers are recorded in JSON.", "",
        "| Batch | Method | Status | Cosine mean ± SD | Normalised MSE mean ± SD | Forward tokens/s | Saved tokens/s | Speedup vs Transformers |", 
        "|---:|---|---|---:|---:|---:|---:|---:|"]
    for directory in sorted(args.output.glob("b[0-9]*"),key=lambda p:int(p.name[1:])):
        comparison_path = directory / "comparison.json"
        comparisons = json.loads(comparison_path.read_text()).get("comparisons",{}) if comparison_path.exists() else {}
        for method in METHODS:
            path = directory / method / "result.json"
            if not path.exists():
                continue
            record = json.loads(path.read_text())
            metrics = record.get("measurements", {})
            value = comparisons.get(method,{})
            agreement = value.get("agreement", {})
            def aggregate(key):
                values = torch.tensor([s[key] for s in agreement.values() if s[key] is not None],dtype=torch.float64)
                return f"{float(values.mean()):.6f} ± {float(values.std(correction=0)):.6f}" if len(values) else "—"
            speedup = value.get("completed_to_disk_speedup")
            rows.append(f"| {record['batch_size']} | {method} | {record['status']} | {aggregate('mean_cosine')} | "
                f"{aggregate('normalized_mse')} | {metrics.get('forward_tokens_per_second',0):.0f} | "
                f"{metrics.get('completed_to_disk_tokens_per_second',0):.0f} | {f'{speedup:.2f}×' if speedup else '—'} |")
    rows += ["", "Means and population standard deviations in the table are across the 18 sites. "
        "Normalised MSE uses the Transformers reference root mean square as a shared scale; it retains magnitude errors. "
        "JSON records all site gates, token cosine standard deviations, actual forward batches, utilisation, errors and stage times. "
        "Different actual forward batches can select different numerical kernels, even with matching requested batches. "
        "The gates remain mean cosine ≥ 0.9999, minimum cosine ≥ 0.9990 and relative L2 ≤ 0.02. "
        "Direct vLLM comparisons use the package as their reference scale. Numerical failures remain failures."]
    (args.output / "comparison.md").write_text("\n".join(rows)+"\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("run","compare","report"))
    parser.add_argument("--method",choices=METHODS)
    parser.add_argument("--work",type=Path,required=True)
    parser.add_argument("--reference",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--batch-size",type=int,choices=(8,16,32,64,128),default=8)
    parser.add_argument("--sequence-length",type=int,default=512)
    parser.add_argument("--tokens",type=int,default=131072)
    parser.add_argument("--prune",action="store_true")
    args = parser.parse_args()
    if args.action == "run" and args.method is None:
        parser.error("run requires --method")
    return {"run":run,"compare":compare,"report":report}[args.action](args)


if __name__ == "__main__":
    raise SystemExit(main())
