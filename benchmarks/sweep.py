# /// script
# requires-python = ">=3.10"
# dependencies = ["nemotron-extraction[hf,vllm,benchmark]==1.0.0"]
# ///
"""Reusable backend phase for a three-language, asynchronous batch-size sweep.

uv run --no-project benchmarks/sweep.py backend --backend hf --batch-size 8 \
    --work outputs/tokens --output outputs/sweep
Run the corresponding vllm phase next, then `report --output outputs/sweep`.
"""
from __future__ import annotations
import argparse
import asyncio
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib.util
import json
import math
import multiprocessing
import os
from pathlib import Path
import tempfile
import time
import uuid
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from nemotron_extraction import HFExtractor, VLLMExtractor, HookSite
from nemotron_extraction.hooks import representative_sites

spec = importlib.util.spec_from_file_location("validate", Path(__file__).with_name("validate.py"))
validate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validate)
SIZES = (8, 16, 32, 64, 128)
TIMINGS = ("model_forward_seconds", "worker_forward_seconds", "cpu_copy_seconds", "worker_copy_seconds",
           "worker_write_seconds", "generate_seconds", "backpressure_seconds", "final_drain_seconds")


def package_digest():
    import nemotron_extraction
    digest = hashlib.sha256()
    for path in sorted(Path(nemotron_extraction.__file__).parent.glob("*.py")):
        digest.update(path.name.encode() + path.read_bytes())
    return digest.hexdigest()


def compare(paths, samples):
    """Dedicated CPU process: at most two queued comparisons, never in a GPU worker."""
    torch.set_num_threads(4)
    started = time.perf_counter()
    reference, candidate = map(load_file, paths)
    if not torch.equal(reference.pop("token_ids"), candidate.pop("token_ids")):
        raise ValueError("Reference and candidate token IDs/order differ")
    if reference.keys() != candidate.keys():
        raise ValueError("Hook sites differ")
    result = {}
    for key in reference:
        metric = validate.Agreement()
        metric.add(reference[key], candidate[key])
        result[key] = (metric.result(), torch.cat(metric.cosines))
        if samples:
            save_file({"hf": reference[key][:8].clone(), "vllm": candidate[key][:8].clone()},
                      str(Path(samples) / f"sample-{key}.safetensors"))
    return result, time.perf_counter() - started


async def language(extractor, args, manifest, language, comparison_pool):
    run = args.output / f"b{args.batch_size}" / language
    run.mkdir(parents=True, exist_ok=True)
    receipt_path = run / f"{args.backend}.json"
    if receipt_path.exists():
        record = json.loads(receipt_path.read_text())
        if (record.get("status") == "complete" and record["tokens"] == args.tokens
                and record["package_sha256"] == package_digest()
                and record["manifest_sha256"] == validate.sha256(args.work / "manifest.json")
                and record["script_sha256"] == validate.sha256(__file__)):
            print(f"resume: {args.backend}/{language}/b{args.batch_size} already complete", flush=True)
            return record
        raise ValueError("Existing incompatible receipt; use a fresh output directory")
    token_path = args.work / f"{language}.safetensors"
    if validate.sha256(token_path) != manifest["languages"][language]["token_sha256"]:
        raise ValueError("Frozen token checksum mismatch")
    tokens = load_file(str(token_path))["token_ids"][:args.tokens]
    output_dir = run / ("reference" if args.backend == "hf" else "candidate")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = run / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    identity = {"package_sha256": package_digest(), "manifest_sha256": validate.sha256(args.work / "manifest.json"),
                "script_sha256": validate.sha256(__file__), "batch_size": args.batch_size, "tokens": args.tokens}
    identity_path = run / f"{args.backend}-identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("Resume identity differs; use a fresh output directory")
    validate.write_json(identity_path, identity)
    before = {key: getattr(extractor, key, 0.) for key in TIMINGS}
    histogram_start = len(extractor.actual_forward_batches)
    count, pending, comparisons, timeline, prefix = 0, [], 0., [], 0
    base_elapsed = 0.
    epoch = uuid.uuid4().hex
    metrics = {site.key: validate.Agreement() for site in extractor.sites}
    hf_run = run.parent / "hf-run.json"
    can_compare = (args.backend == "vllm" and hf_run.exists()
                   and json.loads(hf_run.read_text())["status"] == "complete")
    checkpoints = []
    for index, batch in enumerate(validate.batches(tokens, args.batch_size, 2048)):
        checkpoint = checkpoint_dir / f"{args.backend}-{index:06d}.json"
        if not checkpoint.exists():
            break
        value = json.loads(checkpoint.read_text())
        if value["identity"] != identity or value["tokens"] != batch.numel():
            raise ValueError("Resume shard identity/count differs")
        if args.backend == "hf":
            path = output_dir / f"{index:06d}.safetensors"
            with safe_open(str(path), framework="pt") as saved:
                if not torch.equal(saved.get_tensor("token_ids"), batch):
                    raise ValueError("Resume token order differs")
        elif can_compare:
            cosines = load_file(str(checkpoint.with_suffix(".safetensors")))
            for key, result in value["agreement"].items():
                metric = metrics[key]
                metric.cosines.append(cosines[key])
                metric.error2 += result["error_squared_sum"]
                metric.reference2 += result["reference_squared_sum"]
                metric.max_error = max(metric.max_error, result["maximum_absolute_error"])
            comparisons += value["comparison_seconds"]
        checkpoints.append(value)
        count += value["tokens"]
        base_elapsed = value["pipeline_elapsed"]
        timeline.extend(value["shard"].get("timeline", []))
        prefix += 1
    # Incomplete publications carry no committed checkpoint and can be retried.
    for path in output_dir.glob("*.safetensors*"):
        if args.backend == "vllm" or int(path.name.split(".")[0]) >= prefix:
            path.unlink()
    if prefix:
        print(f"resume {args.backend}/{language}: {count} committed prompt tokens", flush=True)
    started = time.perf_counter()
    last_publication = started

    async def finish(item):
        nonlocal comparisons
        future, paths, shard, global_index = item
        values, seconds = await asyncio.wrap_future(future)
        comparisons += seconds
        for key, (value, cosines) in values.items():
            target = metrics[key]
            target.cosines.append(cosines)
            target.error2 += value["error_squared_sum"]
            target.reference2 += value["reference_squared_sum"]
            target.max_error = max(target.max_error, value["maximum_absolute_error"])
        checkpoint = checkpoint_dir / f"vllm-{global_index:06d}.json"
        save_file({key: cosine for key, (_, cosine) in values.items()}, str(checkpoint.with_suffix(".safetensors")))
        committed = {"identity": identity, "tokens": shard["tokens"], "shard": shard,
                     "comparison_seconds": seconds, "agreement": {key: value for key, (value, _) in values.items()},
                     "pipeline_elapsed": base_elapsed + time.perf_counter() - started}
        validate.write_json(checkpoint, committed)
        checkpoints.append(committed)
        Path(paths[1]).unlink()

    remaining = (batch for index, batch in enumerate(validate.batches(tokens, args.batch_size, 2048)) if index >= prefix)
    # Candidate filenames are relative to the current attempt; reference/checkpoint IDs stay global.
    stream_dir = output_dir if args.backend == "hf" and not prefix else output_dir / f"from-{prefix}"
    stream_dir.mkdir(exist_ok=True)
    for path in stream_dir.glob("*.safetensors*"):
        if stream_dir != output_dir:
            path.unlink()
    stream = extractor.stream_to_disk(remaining, stream_dir)
    try:
        async for shard in stream:
            last_publication = time.perf_counter()
            count += shard["tokens"]
            timeline.extend(shard.get("timeline", []))
            for item in shard.get("timeline", []):
                item["attempt"] = epoch
            global_index = prefix + shard["index"]
            if args.backend == "hf":
                canonical = output_dir / f"{global_index:06d}.safetensors"
                if Path(shard["path"]) != canonical:
                    Path(shard["path"]).replace(canonical)
                    shard["path"] = str(canonical)
                committed = {"identity": identity, "tokens": shard["tokens"], "shard": shard,
                             "pipeline_elapsed": base_elapsed + last_publication - started}
                validate.write_json(checkpoint_dir / f"hf-{global_index:06d}.json", committed)
                checkpoints.append(committed)
            if can_compare:
                reference = run / "reference" / f"{global_index:06d}.safetensors"
                if not reference.exists():
                    raise ValueError(f"Missing HF shard {reference.name}")
                paths = (str(reference), shard["path"])
                samples = str(run) if global_index == 0 else None
                pending.append((comparison_pool.submit(compare, paths, samples), paths, shard, global_index))
                if len(pending) >= 2:
                    await finish(pending.pop(0))
            elif args.backend == "vllm":
                committed = {"identity": identity, "tokens": shard["tokens"], "shard": shard,
                             "pipeline_elapsed": base_elapsed + last_publication - started}
                validate.write_json(checkpoint_dir / f"vllm-{global_index:06d}.json", committed)
                checkpoints.append(committed)
            if shard["index"] % 10 == 0:
                print(f"{args.backend}/{language}/b{args.batch_size}: {count}/{args.tokens}", flush=True)
    finally:
        await stream.aclose()
    publication_seconds = base_elapsed + last_publication - started
    for item in pending:
        await finish(item)
    elapsed = base_elapsed + time.perf_counter() - started
    if count != args.tokens:
        raise ValueError(f"Expected {args.tokens} prompt tokens, got {count}")
    timings = {key: getattr(extractor, key, 0.) - before[key] for key in TIMINGS}
    # GPU event intervals remain exact when a phase continues after a job deadline.
    event_seconds = lambda name: sum(item["end"] - item["start"] for item in timeline if item["name"] == name)
    timings["model_forward_seconds" if args.backend == "hf" else "worker_forward_seconds"] = event_seconds("forward")
    timings["cpu_copy_seconds" if args.backend == "hf" else "worker_copy_seconds"] = event_seconds("D2H")
    timings["worker_write_seconds"] = sum(item["shard"]["write_seconds"] for item in checkpoints)
    forward = timings["model_forward_seconds"] if args.backend == "hf" else timings["worker_forward_seconds"]
    record = {"status": "complete", "backend": args.backend, "language": language,
              "batch_size": args.batch_size, "tokens": count, "sequence_length": 2048,
              "publication_seconds": publication_seconds, "phase_seconds": elapsed,
              "forward_tokens_per_second": count / forward if forward else None,
              "disk_tokens_per_second": count / publication_seconds,
              "comparison_seconds": comparisons, **timings,
              "resumed_shards": prefix, "timing_scope": "all committed shards; admission/generation/drain counters cover current attempt",
              "actual_forward_batches": dict(Counter(
                  size for checkpoint in checkpoints for size in checkpoint["shard"].get("actual_forward_batches",
                      [min(args.batch_size, max(1, checkpoint["tokens"] // 2048))]))),
              "topology": extractor.topology, "software": validate.versions(),
              "gpu": torch.cuda.get_device_name(), "manifest_sha256": validate.sha256(args.work / "manifest.json"),
              "package_sha256": package_digest(), "script_sha256": validate.sha256(__file__),
              "source": {"model_revision": manifest["model_revision"], "tokenizer_revision": manifest["tokenizer_revision"],
                         "dataset": manifest["languages"][language]}, "cuda_graphs_enabled": False,
              "memory_limit_fraction": .96, "kv_cache_memory_bytes": getattr(extractor, "kv_cache_memory_bytes", None),
              "peak_gpu_memory_bytes": extractor.peak_gpu_memory_bytes,
              "peak_gpu_reserved_bytes": extractor.peak_gpu_reserved_bytes,
              "peak_device_memory_bytes": extractor.peak_device_memory_bytes}
    forwards = [item for item in timeline if item["name"] == "forward"]
    record["overlap_seconds"] = {name: sum(max(0., min(a["end"], b["end"]) - max(a["start"], b["start"]))
                                             for a in forwards for b in timeline
                                             if b["name"] == name and a.get("attempt") == b.get("attempt"))
                                 for name in ("D2H", "write")}
    if can_compare:
        record["agreement"] = {key: metric.result() for key, metric in metrics.items()}
        record["passed"] = all(metric["passed"] for metric in record["agreement"].values())
        record["validation_status"] = "pass" if record["passed"] else "fail"
        save_file({key: torch.cat(metric.cosines) for key, metric in metrics.items()}, str(run / "cosines.safetensors"))
    elif args.backend == "vllm":
        record.update(agreement={}, passed=False, validation_status="unavailable: HF phase did not complete")
        for path in output_dir.glob("*.safetensors"):
            path.unlink()
    if args.profile:
        validate.write_json(run / f"{args.backend}-trace.json", {"traceEvents": [
            {"name": item["name"], "ph": "X", "pid": args.backend, "tid": item["name"],
             "ts": item["start"] * 1e6, "dur": (item["end"] - item["start"]) * 1e6,
             "args": {"attempt": item.get("attempt")}} for item in timeline]})
    validate.write_json(receipt_path, record)
    if can_compare:
        # Baseline deletion follows durable language metrics, never an uncommitted comparison.
        for path in (run / "reference").glob("*.safetensors"):
            path.unlink()
    return record


def backend(args):
    torch.set_num_threads(4)
    manifest = json.loads((args.work / "manifest.json").read_text())
    root = args.output / f"b{args.batch_size}"
    root.mkdir(parents=True, exist_ok=True)
    sites_file = args.output / "sites.json"
    started = time.perf_counter()
    extractor = None
    records = []
    phase = "startup"
    try:
        if args.backend == "hf":
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(manifest["model_path"], torch_dtype=torch.bfloat16,
                                                        device_map="cuda", trust_remote_code=True, local_files_only=True)
            model_load = time.perf_counter() - started
            sites = representative_sites(model)
            extractor = HFExtractor(model, sites)
            validate.write_json(sites_file, [{"layer": site.layer, "kind": site.kind} for site in sites])
        else:
            sites = [HookSite(**site) for site in json.loads(sites_file.read_text())]
            extractor = VLLMExtractor(manifest["model_path"], sites, sequences_per_call=args.batch_size,
                                      sink_dir=os.environ.get("NH_SINK_DIR"), max_model_len=2049,
                                      gpu_memory_utilization=.96,
                                      kv_cache_memory_bytes=int(os.environ.get("NH_KV_CACHE_BYTES", 1536 * 2**20)))
            model_load = extractor.engine_startup_seconds
        startup = time.perf_counter() - started
        phase = "warmup"
        tokens = load_file(str(args.work / "en.safetensors"))["token_ids"]
        warm_tokens = tokens[:args.batch_size * 2048].reshape(args.batch_size, 2048)
        warm_started = time.perf_counter()
        # Two complete batches exercise both reusable capture slots.
        async def warmup():
            with tempfile.TemporaryDirectory(dir=os.environ.get("NH_SINK_DIR")) as directory:
                traces = []
                async for receipt in extractor.stream_to_disk([warm_tokens, warm_tokens], directory):
                    traces.extend(receipt.get("timeline", []))
                if args.profile:
                    validate.write_json(root / f"{args.backend}-warmup-trace.json", {"traceEvents": [
                        {"name": item["name"], "ph": "X", "pid": args.backend, "tid": item["name"],
                         "ts": item["start"] * 1e6, "dur": (item["end"] - item["start"]) * 1e6} for item in traces]})
        asyncio.run(warmup())
        warmup_seconds = time.perf_counter() - warm_started
        phase = "extraction"
        with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as pool:
            for lang in ("en", "de", "es"):
                records.append(asyncio.run(language(extractor, args, manifest, lang, pool)))
        summary = {"status": "complete", "startup_seconds": startup, "model_load_seconds": model_load,
                   "hook_install_seconds": extractor.hook_install_seconds, "warmup_seconds": warmup_seconds,
                   "phase_seconds": time.perf_counter() - started, "languages": records}
    except Exception as error:
        message = str(error)
        memory_failure = isinstance(error, torch.OutOfMemoryError) or any(
            text in message.lower() for text in ("out of memory", "gpu memory budget exceeded", "no available memory"))
        summary = {"status": "memory_failure" if memory_failure else "error", "failed_phase": phase,
                   "error_type": type(error).__name__, "error": message,
                   "phase_seconds": time.perf_counter() - started, "languages": records}
        validate.write_json(root / f"{args.backend}-run.json", summary)
        if not memory_failure:
            raise
    finally:
        if extractor is not None:
            try:
                extractor.close()
            except Exception as cleanup_error:
                if "summary" not in locals() or summary["status"] == "complete":
                    raise
                summary["cleanup_error"] = str(cleanup_error)
    summary.update(backend=args.backend, batch_size=args.batch_size, requested_tokens=args.tokens,
                   package_sha256=package_digest(), software=validate.versions())
    validate.write_json(root / f"{args.backend}-run.json", summary)
    print(json.dumps({key: summary[key] for key in ("status", "backend", "batch_size", "phase_seconds")}), flush=True)


def report(args):
    rows = []
    for batch in SIZES:
        for lang in ("en", "de", "es"):
            pair = []
            for engine in ("hf", "vllm"):
                path = args.output / f"b{batch}" / lang / f"{engine}.json"
                run_path = args.output / f"b{batch}" / f"{engine}-run.json"
                pair.append(json.loads(path.read_text()) if path.exists() else
                            json.loads(run_path.read_text()) if run_path.exists() else {"status": "pending"})
            hf, vllm = pair
            if hf["status"] == vllm["status"] == "complete" and "tokens" in hf and "tokens" in vllm:
                for key in ("tokens", "batch_size", "manifest_sha256", "package_sha256", "script_sha256", "source"):
                    if hf[key] != vllm[key]:
                        raise ValueError(f"Backend receipts disagree on {key}")
                speedup = vllm["disk_tokens_per_second"] / hf["disk_tokens_per_second"]
                rows.append({"batch": batch, "language": lang, "hf": hf, "vllm": vllm, "disk_speedup": speedup})
            else:
                rows.append({"batch": batch, "language": lang, "hf": hf, "vllm": vllm})
    validate.write_json(args.output / "results.json", rows)
    lines = ["# Asynchronous activation batch sweep", "", "| Batch | Language | HF disk tok/s | vLLM disk tok/s | Ratio | Gate/status |",
             "|---:|---|---:|---:|---:|---|"]
    for row in rows:
        if "disk_speedup" in row:
            lines.append(f"| {row['batch']} | {row['language']} | {row['hf']['disk_tokens_per_second']:.1f} | {row['vllm']['disk_tokens_per_second']:.1f} | {row['disk_speedup']:.2f} | {'PASS' if row['vllm']['passed'] else 'FAIL'} |")
        else:
            rates = [f"{row[engine]['disk_tokens_per_second']:.1f}" if "disk_tokens_per_second" in row[engine] else "—"
                     for engine in ("hf", "vllm")]
            lines.append(f"| {row['batch']} | {row['language']} | {rates[0]} | {rates[1]} | — | HF: {row['hf']['status']}; vLLM: {row['vllm']['status']} |")
    lines += ["", "Disk throughput includes capture, transfer, publication and pipeline backpressure. Numerical comparison runs in a separate CPU process and can limit admission when its bounded queue fills. Forward throughput, actual forward sizes, startup, warmup, copy/write timings and memory peaks are in JSON. CUDA graphs are disabled. A numerical failure remains a failure regardless of throughput."]
    (args.output / "report.md").write_text("\n".join(lines) + "\n")


def schedule(args):
    jobs = []
    for batch in SIZES:
        total = 0.
        for engine in ("hf", "vllm"):
            path = args.output / f"b{batch}" / f"{engine}-run.json"
            if not path.exists():
                raise ValueError(f"Missing pilot receipt: b{batch}/{engine}")
            record = json.loads(path.read_text())
            if record["status"] == "memory_failure":
                total += record["phase_seconds"]
                continue
            if record["status"] != "complete" or len(record["languages"]) != 3:
                raise ValueError("Pilot must cover all languages or document a memory failure")
            total += record["startup_seconds"] + record["warmup_seconds"]
            total += sum(lang["phase_seconds"] / lang["tokens"] * 1_000_000 for lang in record["languages"])
        jobs.append({"batch_size": batch, "estimated_seconds": total,
                     "minutes": math.ceil((total * 1.25 + 120) / 300) * 5})
    minutes = math.ceil((sum(job["estimated_seconds"] for job in jobs) * 1.25 + 120) / 300) * 5
    print(json.dumps({"combined_minutes": minutes, "fits_one_job": minutes <= 60, "batch_jobs": jobs}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("backend", "report", "schedule"):
        child = sub.add_parser(name)
        child.add_argument("--output", type=Path, required=True)
        if name == "backend":
            child.add_argument("--work", type=Path, required=True)
            child.add_argument("--backend", choices=("hf", "vllm"), required=True)
            child.add_argument("--batch-size", type=int, choices=SIZES, required=True)
            child.add_argument("--tokens", type=int, choices=(16384, 1_000_000), default=1_000_000)
            child.add_argument("--profile", action="store_true", help="Write CUDA-event/write overlap traces")
    args = parser.parse_args()
    {"backend": backend, "report": report, "schedule": schedule}[args.command](args)


if __name__ == "__main__":
    main()
