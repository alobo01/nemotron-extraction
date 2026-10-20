# /// script
# requires-python = ">=3.10"
# dependencies = ["nemotron-extraction[hf,vllm,benchmark]==1.1.0"]
# ///
"""Compare both backends on frozen FineWeb inputs for English, German and Spanish.

uv run benchmarks/validate.py all --work outputs/tokens --output outputs/validation

When one job cannot finish a language, run a pilot with `run --tokens 16384` into
<pilot>/<lang>-0, split the full run with `plan --pilot <pilot>`, run each range with
`run --start/--tokens`, and join the ranges with `aggregate`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
from safetensors.torch import load_file, save_file
from nemotron_extraction import HFExtractor, VLLMExtractor, HookSite
from nemotron_extraction.hooks import representative_sites

from benchlib.common import (DATA, MODEL, MODEL_REVISION, batches, digest as sha256, job_minutes,
                             load_tokens, versions, write_json)
from benchlib.stats import Agreement, normalized_mse

TOKENS_PER_LANGUAGE = 1_000_000
BATCH_SIZES = (1, 2, 4, 8, 16, 32, 64, 128)


def prepare(args):
    from datasets import load_dataset
    from huggingface_hub import HfApi, snapshot_download
    from transformers import AutoTokenizer
    root = args.work
    root.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    revision = api.model_info(MODEL, revision=args.revision).sha
    model = snapshot_download(MODEL, revision=revision,
                              allow_patterns=["*.json", "*.py", "*.safetensors", "*.txt", "*.model"])
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    if tokenizer.eos_token_id is None:
        raise ValueError("An EOS token is required for document boundaries")
    manifest = {"model": MODEL, "model_revision": revision, "model_path": model,
                "tokenizer_revision": revision, "sequence_length": 2048, "tokens_per_language": TOKENS_PER_LANGUAGE,
                "dtype": "bfloat16", "seed": 42, "document_order": "dataset iteration order; no shuffle",
                "languages": {}, "software": versions()}
    for language, (dataset, config) in DATA.items():
        data_revision = api.dataset_info(dataset).sha
        stream = load_dataset(dataset, config, revision=data_revision, split="train", streaming=True)
        tokens, documents, order = [], 0, hashlib.sha256()
        for row in stream:
            if not row.get("text"):
                continue
            encoded = tokenizer(row["text"], add_special_tokens=False)["input_ids"]
            tokens.extend(encoded + [tokenizer.eos_token_id])
            order.update((str(row.get("id", "")) + "\0" + row["text"]).encode())
            documents += 1
            if len(tokens) >= TOKENS_PER_LANGUAGE:
                break
        if len(tokens) < TOKENS_PER_LANGUAGE:
            raise ValueError(f"Insufficient {language} tokens")
        path = root / f"{language}.safetensors"
        save_file({"token_ids": torch.tensor(tokens[:TOKENS_PER_LANGUAGE], dtype=torch.long)}, str(path))
        manifest["languages"][language] = {"dataset": dataset, "config": config, "revision": data_revision,
                                           "split": "train", "documents": documents,
                                           "document_order_sha256": order.hexdigest(), "token_sha256": sha256(path)}
        print(f"prepared {language}: 1,000,000 tokens", flush=True)
    write_json(root / "manifest.json", manifest)


def backend(args):
    torch.set_num_threads(min(16, os.cpu_count() or 1))
    manifest = json.loads((args.work / "manifest.json").read_text())
    data = manifest["languages"][args.language]
    all_tokens = load_tokens(args.work, args.language, manifest)
    stop = min(args.start + args.tokens, all_tokens.numel())
    if args.start < 0 or args.start % 2048 or args.start >= stop or stop - args.start != args.tokens:
        raise ValueError("Invalid token range")
    tokens = all_tokens[args.start:stop]
    run = args.output
    reference_dir = run / "reference"
    reference_dir.mkdir(parents=True, exist_ok=True)
    sites_file = run / "sites.json"
    started = time.perf_counter()
    if args.backend == "hf":
        from transformers import AutoModelForCausalLM
        phase = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(manifest["model_path"], torch_dtype=torch.bfloat16,
                                                   device_map="cuda", trust_remote_code=True, local_files_only=True)
        model_load_seconds = time.perf_counter() - phase
        phase = time.perf_counter()
        sites = representative_sites(model)
        extractor = HFExtractor(model, sites)
        hook_install_seconds = time.perf_counter() - phase
        write_json(sites_file, [{"layer": site.layer, "kind": site.kind} for site in sites])
    else:
        sites = [HookSite(**site) for site in json.loads(sites_file.read_text())]
        extractor = VLLMExtractor(manifest["model_path"], sites, sequences_per_call=args.batch_size,
                                  max_model_len=2049, sink_dir=os.environ.get("NH_SINK_DIR"),
                                  gpu_memory_utilization=float(os.environ.get("NH_GPU_MEMORY_UTILIZATION", "0.96")),
                                  kv_cache_memory_bytes=None)
        model_load_seconds = extractor.engine_startup_seconds
        hook_install_seconds = extractor.hook_install_seconds
    startup = model_load_seconds + hook_install_seconds
    print(f"{args.backend} ready: model/engine {model_load_seconds:.1f}s, hooks {hook_install_seconds:.1f}s", flush=True)
    extraction, comparison, persistence, count = 0., 0., 0., 0
    stats = {site.key: Agreement() for site in sites}
    torch.cuda.reset_peak_memory_stats()
    with extractor:
        # Warmup has no effect on counted tokens or recorded samples.
        torch.cuda.synchronize()
        phase = time.perf_counter()
        extractor.extract(tokens[:min(128, len(tokens))].unsqueeze(0))
        torch.cuda.synchronize()
        warmup_seconds = time.perf_counter() - phase
        timing_names = ("model_forward_seconds", "cpu_copy_seconds", "generate_seconds",
                        "worker_collection_seconds", "driver_read_seconds", "worker_copy_seconds",
                        "worker_write_seconds")
        timing_before = {name: getattr(extractor, name, 0.0) for name in timing_names}
        for index, batch in enumerate(batches(tokens, args.batch_size, 2048)):
            torch.cuda.synchronize()
            begin = time.perf_counter()
            result = extractor.extract(batch)
            torch.cuda.synchronize()
            extraction += time.perf_counter() - begin
            if not torch.equal(result.token_ids, batch):
                raise ValueError("Extractor changed input token order")
            shard = reference_dir / f"{index:06d}.safetensors"
            begin = time.perf_counter()
            if args.backend == "hf":
                save_file({"token_ids": batch, **result.activations}, str(shard))
                persistence += time.perf_counter() - begin
            else:
                reference = load_file(str(shard))
                if not torch.equal(reference.pop("token_ids"), batch):
                    raise ValueError("Reference token IDs differ")
                if set(reference) != set(result.activations):
                    raise ValueError("Reference hook sites differ")
                for key, value in result.activations.items():
                    stats[key].add(reference[key], value)
                    if index == 0:
                        save_file({"hf": reference[key][:8].clone(), "vllm": value[:8].clone()}, str(run / f"sample-{key}.safetensors"))
                del reference
                comparison += time.perf_counter() - begin
            count += batch.numel()
            if index % 10 == 0:
                print(f"{args.backend}: {count}/{args.tokens} tokens", flush=True)
        topology = extractor.topology
        peak = max(torch.cuda.max_memory_allocated(), getattr(extractor, "peak_gpu_memory_bytes", 0))
    receipt = {"backend": args.backend, "language": args.language, "tokens": count,
               "start_token": args.start, "batch_size": args.batch_size, "sequence_length": 2048,
               "model_load_seconds": model_load_seconds, "hook_install_seconds": hook_install_seconds,
               "startup_seconds": startup, "warmup_seconds": warmup_seconds,
               "inference_seconds": (getattr(extractor, "model_forward_seconds", 0.0) - timing_before["model_forward_seconds"]
                                     if args.backend == "hf" else extractor.generate_seconds - timing_before["generate_seconds"]),
               "capture_copy_seconds": (getattr(extractor, "cpu_copy_seconds", 0.0) - timing_before["cpu_copy_seconds"]
                                        if args.backend == "hf" else extractor.worker_copy_seconds - timing_before["worker_copy_seconds"]),
               "worker_write_seconds": (extractor.worker_write_seconds - timing_before["worker_write_seconds"]
                                        if args.backend == "vllm" else 0.0),
               "worker_collection_seconds": (extractor.worker_collection_seconds - timing_before["worker_collection_seconds"]
                                              if args.backend == "vllm" else 0.0),
               "driver_read_seconds": getattr(extractor, "driver_read_seconds", 0.0) - timing_before["driver_read_seconds"],
               "extraction_seconds": extraction,
               "comparison_seconds": comparison, "persistence_seconds": persistence,
               "wall_seconds": time.perf_counter() - started, "tokens_per_second": count / extraction,
               "peak_gpu_memory_bytes": peak, "topology": topology, "software": versions(),
               "gpu_memory_utilization": (extractor.gpu_memory_utilization if args.backend == "vllm" else None),
               "kv_cache_memory_bytes": (extractor.kv_cache_memory_bytes if args.backend == "vllm" else None),
               "cuda_graphs_enabled": False,
               "gpu": torch.cuda.get_device_name(),
               "source": {"model": manifest["model"], "model_revision": manifest["model_revision"],
                          "tokenizer_revision": manifest["tokenizer_revision"], "dataset": data,
                          "dtype": manifest["dtype"]},
               "manifest_sha256": sha256(args.work / "manifest.json"),
               "script_sha256": sha256(__file__)}
    if args.backend == "vllm":
        receipt["agreement"] = {key: value.result() for key, value in stats.items()}
        receipt["passed"] = all(value["passed"] for value in receipt["agreement"].values())
        save_file({key: torch.cat(value.cosines) for key, value in stats.items()}, str(run / "cosines.safetensors"))
    write_json(run / f"{args.backend}.json", receipt)
    # Delete only this run's raw baseline after durable comparison metrics exist.
    if args.backend == "vllm":
        for shard in reference_dir.glob("*.safetensors"):
            shard.unlink()
    print(json.dumps({k: receipt[k] for k in ("backend", "tokens", "tokens_per_second")}), flush=True)


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "vllm.json").exists() or (args.output / "hf.json").exists():
        raise ValueError("Use a new output directory; completed receipts are immutable")
    common = ["--work", str(args.work), "--output", str(args.output), "--language", args.language,
              "--tokens", str(args.tokens), "--start", str(args.start), "--batch-size", str(args.batch_size)]
    for engine in ("hf", "vllm"):
        command = [sys.executable, str(Path(__file__).resolve()), "backend", "--backend", engine, *common]
        if args.hf_python and engine == "hf":
            command[0] = args.hf_python
        result = subprocess.run(command)
        if result.returncode:
            raise SystemExit(result.returncode)
    report(args)


def report(args):
    hf = json.loads((args.output / "hf.json").read_text())
    vllm = json.loads((args.output / "vllm.json").read_text())
    for key in ("language", "tokens", "batch_size", "start_token", "manifest_sha256",
                "sequence_length", "script_sha256", "source", "gpu"):
        if hf[key] != vllm[key]:
            raise ValueError(f"Backend receipts disagree on {key}")
    result = {"hf": hf, "vllm": vllm, "passed": vllm["passed"], "speedup": vllm["tokens_per_second"] / hf["tokens_per_second"]}
    write_json(args.output / "results.json", result)
    lines = [f"# {hf['language']} activation validation", "", f"Tokens: {hf['tokens']:,}. Numerical gate: {'PASS' if result['passed'] else 'FAIL'}.", "",
             "| Backend | Load s | Hooks s | Warmup s | Inference s | H2C copy s | Sink write s | Collect/read s | Full tok/s | Peak GiB |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for receipt in (hf, vllm):
        lines.append(f"| {receipt['backend']} | {receipt['model_load_seconds']:.1f} | {receipt['hook_install_seconds']:.1f} | {receipt['warmup_seconds']:.1f} | {receipt['inference_seconds']:.1f} | {receipt['capture_copy_seconds']:.1f} | {receipt['worker_write_seconds']:.1f} | {receipt['worker_collection_seconds'] + receipt['driver_read_seconds']:.1f} | {receipt['tokens_per_second']:.1f} | {receipt['peak_gpu_memory_bytes'] / 2**30:.2f} |")
    lines += ["", "Inference excludes model loading, hook setup, warmup and post-generation collection. GPU-to-host copies are shown separately; vLLM performs them inside generation. Sink writes are asynchronous and may overlap inference. Full extraction throughput measures the complete extractor call. CUDA graphs are disabled by eager execution.", "",
              "Normalised mean squared error (MSE) scales both backends by the Hugging Face Transformers reference root mean square at each site. It equals summed squared error divided by the summed squared reference, or relative L2 squared. Numerical thresholds are unchanged.", "",
              "| Site | Mean cosine | Minimum cosine | p01 cosine | Normalised MSE | Relative L2 | Max error | Gate |", "|---|---:|---:|---:|---:|---:|---:|---|"]
    for key, value in vllm["agreement"].items():
        nmse = normalized_mse(value['error_squared_sum'], value['reference_squared_sum'])
        lines.append(f"| {key} | {value['mean_cosine']:.6f} | {value['minimum_cosine']:.6f} | {value['first_percentile_cosine']:.6f} | {nmse} | {value['relative_l2']} | {value['maximum_absolute_error']:.5f} | {'PASS' if value['passed'] else 'FAIL'} |")
    (args.output / "report.md").write_text("\n".join(lines) + "\n")


def run_all(args):
    """Stage once, then run languages and backends sequentially."""
    for language in DATA:
        output = args.output / language
        if (output / "hf.json").exists() != (output / "vllm.json").exists():
            raise ValueError(f"Incomplete backend receipts in {output}; use a new output directory")
    if not (args.work / "manifest.json").exists():
        prepare(args)
    manifest = json.loads((args.work / "manifest.json").read_text())
    if manifest["model"] != MODEL or manifest["model_revision"] != args.revision:
        raise ValueError("Cached model revision differs from the requested revision")
    for language in DATA:
        output = args.output / language
        run_args = argparse.Namespace(
            work=args.work, output=output, language=language, tokens=TOKENS_PER_LANGUAGE,
            start=0, batch_size=args.batch_size, hf_python=args.hf_python,
        )
        if (output / "hf.json").exists() and (output / "vllm.json").exists():
            for engine in ("hf", "vllm"):
                receipt = json.loads((output / f"{engine}.json").read_text())
                expected = {"language": language, "tokens": TOKENS_PER_LANGUAGE, "start_token": 0,
                            "batch_size": args.batch_size,
                            "manifest_sha256": sha256(args.work / "manifest.json")}
                if any(receipt.get(key) != value for key, value in expected.items()):
                    raise ValueError(f"Completed receipts in {output} differ from this request")
            report(run_args)
        else:
            run(run_args)


def plan_ranges(pilot, batch_size):
    """Split each language into contiguous ranges that fit one-hour jobs, from pilot receipts."""
    jobs = []
    records = {}
    for language in ("en", "de", "es"):
        path = pilot / f"{language}-0"
        hf = json.loads((path / "hf.json").read_text())
        vllm = json.loads((path / "vllm.json").read_text())
        if hf["tokens"] != 16384 or vllm["tokens"] != 16384:
            raise ValueError("Need complete 16,384-token pilot receipts")
        if hf["batch_size"] != batch_size or vllm["batch_size"] != batch_size:
            raise ValueError("Both pilot backends must use the selected batch size")
        records[language] = (hf, vllm)
    # A new allocation can have cold weights even if later pilot languages were warm.
    startup = max(pair[0]["startup_seconds"] for pair in records.values())
    startup += max(pair[1]["startup_seconds"] for pair in records.values())
    for language, (hf, vllm) in records.items():
        # Wall minus startup includes warmup, cleanup, persistence and comparison.
        fixed = max(0, hf["wall_seconds"] - hf["startup_seconds"] - hf["extraction_seconds"] - hf["persistence_seconds"])
        fixed += max(0, vllm["wall_seconds"] - vllm["startup_seconds"] - vllm["extraction_seconds"] - vllm["comparison_seconds"])
        per_token = (hf["extraction_seconds"] + hf["persistence_seconds"] +
                     vllm["extraction_seconds"] + vllm["comparison_seconds"]) / 16384
        available = (3600 - 120) / 1.25 - startup - fixed
        maximum = int(available / per_token / 2048) * 2048
        if maximum < 2048:
            raise ValueError("Model startup leaves no useful work in a one-hour job")
        start = 0
        while start < TOKENS_PER_LANGUAGE:
            count = min(maximum, TOKENS_PER_LANGUAGE - start)
            minutes = job_minutes(startup + fixed + count * per_token)
            jobs.append({"language": language, "start": start, "tokens": count,
                         "minutes": min(60, minutes), "batch_size": batch_size})
            start += count
    return jobs


def aggregate_ranges(paths, output, expected=TOKENS_PER_LANGUAGE):
    """Join contiguous range receipts into one language receipt and report."""
    receipts = []
    for path in paths:
        hf = json.loads((path / "hf.json").read_text())
        vllm = json.loads((path / "vllm.json").read_text())
        report(argparse.Namespace(output=path))
        receipts.append((hf["start_token"], path, hf, vllm))
    receipts.sort()
    first = receipts[0]
    end = 0
    combined = {}
    for start, path, hf, vllm in receipts:
        if start != end:
            raise ValueError("Ranges must be contiguous, non-overlapping and begin at zero")
        for engine, record, baseline in (("hf", hf, first[2]), ("vllm", vllm, first[3])):
            for key in ("language", "manifest_sha256", "batch_size", "sequence_length", "topology", "software", "script_sha256", "source", "gpu", "gpu_memory_utilization", "kv_cache_memory_bytes", "cuda_graphs_enabled"):
                if record[key] != baseline[key]:
                    raise ValueError(f"Ranges disagree on {engine} {key}")
        end += hf["tokens"]
        cosines = load_file(str(path / "cosines.safetensors"))
        for key, metric in vllm["agreement"].items():
            if cosines[key].numel() != hf["tokens"]:
                raise ValueError("Cosine rows do not match counted tokens")
            combined.setdefault(key, Agreement()).merge_result(metric, cosines[key])
    if end != expected:
        raise ValueError(f"Expected exactly {expected:,} tokens, got {end:,}")
    output.mkdir(parents=True, exist_ok=True)
    for engine, column in (("hf", 2), ("vllm", 3)):
        result = dict(first[column])
        result.update(tokens=end, start_token=0, ranges=len(receipts))
        for key in ("model_load_seconds", "hook_install_seconds", "startup_seconds", "warmup_seconds", "inference_seconds", "capture_copy_seconds", "driver_read_seconds", "extraction_seconds", "comparison_seconds", "persistence_seconds", "wall_seconds"):
            result[key] = sum(row[column][key] for row in receipts)
        result["tokens_per_second"] = end / result["extraction_seconds"]
        result["peak_gpu_memory_bytes"] = max(row[column]["peak_gpu_memory_bytes"] for row in receipts)
        if engine == "vllm":
            result["agreement"] = {key: value.result() for key, value in combined.items()}
            result["passed"] = all(value["passed"] for value in result["agreement"].values())
        write_json(output / f"{engine}.json", result)
    report(argparse.Namespace(output=output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    stage = sub.add_parser("prepare", help="CPU stage: pin revisions, cache public weights, tokenize once")
    stage.add_argument("--work", type=Path, required=True)
    stage.add_argument("--revision", default=MODEL_REVISION)
    complete = sub.add_parser("all", help="Prepare inputs and compare all three languages sequentially")
    complete.add_argument("--work", type=Path, default=Path("outputs/tokens"))
    complete.add_argument("--output", type=Path, default=Path("outputs/validation"))
    complete.add_argument("--revision", default=MODEL_REVISION)
    complete.add_argument("--batch-size", type=int, choices=BATCH_SIZES, default=4)
    complete.add_argument("--hf-python", help="Compatible Hugging Face Transformers Python, if separate")
    for command in ("run", "backend", "report"):
        child = sub.add_parser(command)
        child.add_argument("--output", type=Path, required=True)
        if command == "report":
            continue
        child.add_argument("--work", type=Path, required=True)
        child.add_argument("--language", choices=DATA, required=True)
        child.add_argument("--tokens", type=int, default=TOKENS_PER_LANGUAGE)
        child.add_argument("--start", type=int, default=0)
        child.add_argument("--batch-size", type=int, choices=BATCH_SIZES, default=4)
        if command == "backend":
            child.add_argument("--backend", choices=("hf", "vllm"), required=True)
        else:
            child.add_argument("--hf-python", help="Existing compatible HF environment, if different from vLLM")
    planning = sub.add_parser("plan", help="Print job ranges from <pilot>/<lang>-0 receipts; submits nothing")
    planning.add_argument("--pilot", type=Path, required=True)
    planning.add_argument("--batch-size", type=int, choices=BATCH_SIZES, default=4)
    joining = sub.add_parser("aggregate", help="Join contiguous range receipts for one language")
    joining.add_argument("--ranges", nargs="+", type=Path, required=True)
    joining.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "plan":
        print(json.dumps(plan_ranges(args.pilot, args.batch_size), indent=2))
    elif args.command == "aggregate":
        aggregate_ranges(args.ranges, args.output)
    else:
        {"prepare": prepare, "backend": backend, "run": run, "report": report, "all": run_all}[args.command](args)


if __name__ == "__main__":
    main()
