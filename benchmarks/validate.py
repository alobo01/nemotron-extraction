# /// script
# requires-python = ">=3.10"
# dependencies = ["nemotron-extraction[hf,vllm,benchmark]==1.0.0"]
# ///
"""Single-file benchmark: prepare frozen inputs and compare both backends.

uv run benchmarks/validate.py all --work outputs/tokens --output outputs/validation
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
from safetensors.torch import load_file, save_file
from nemotron_extraction import HFExtractor, VLLMExtractor, HookSite
from nemotron_extraction.hooks import representative_sites

MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
MODEL_REVISION = "fe4143e388795a58a92974891b2e45092a53dc1d"
DATA = {"en": ("HuggingFaceFW/fineweb-edu", "sample-10BT"),
        "de": ("HuggingFaceFW/fineweb-2", "deu_Latn"),
        "es": ("HuggingFaceFW/fineweb-2", "spa_Latn")}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def versions():
    result = {"python": sys.version.split()[0], "cuda": torch.version.cuda}
    for name in ("nemotron-extraction", "torch", "transformers", "vllm", "datasets", "safetensors",
                 "mamba-ssm", "causal-conv1d", "accelerate", "kernels"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def batches(tokens, batch_size, seq_len):
    full = tokens.numel() // seq_len * seq_len
    if full:
        for chunk in tokens[:full].reshape(-1, seq_len).split(batch_size):
            yield chunk
    if full < tokens.numel():
        yield tokens[full:].unsqueeze(0)


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
                "tokenizer_revision": revision, "sequence_length": 2048, "tokens_per_language": 1_000_000,
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
            if len(tokens) >= 1_000_000:
                break
        if len(tokens) < 1_000_000:
            raise ValueError(f"Insufficient {language} tokens")
        path = root / f"{language}.safetensors"
        save_file({"token_ids": torch.tensor(tokens[:1_000_000], dtype=torch.long)}, str(path))
        manifest["languages"][language] = {"dataset": dataset, "config": config, "revision": data_revision,
                                           "split": "train", "documents": documents,
                                           "document_order_sha256": order.hexdigest(), "token_sha256": sha256(path)}
        print(f"prepared {language}: 1,000,000 tokens", flush=True)
    write_json(root / "manifest.json", manifest)


def normalized_mse(error_squared_sum, reference_squared_sum):
    """MSE after scaling both tensors by the reference's site-wide RMS.

    The shared scale preserves magnitude errors. A zero reference has no
    usable scale: return zero for identical zeros and None otherwise.
    """
    if reference_squared_sum:
        return error_squared_sum / reference_squared_sum
    return 0. if not error_squared_sum else None


class Agreement:
    """FP32 differences and norms, accumulated in float64; exact cosine quantiles."""
    def __init__(self):
        self.cosines, self.error2, self.reference2, self.max_error = [], 0., 0., 0.

    def add(self, reference, candidate):
        if reference.shape != candidate.shape or reference.dtype != candidate.dtype:
            raise ValueError("Activation shapes or dtypes differ")
        for left, right in zip(reference.split(256), candidate.split(256)):
            left, right = left.float(), right.float()
            if not bool(torch.isfinite(left).all() and torch.isfinite(right).all()):
                raise ValueError("Nonfinite activations")
            diff = right - left
            self.error2 += float(diff.double().square().sum())
            self.reference2 += float(left.double().square().sum())
            self.max_error = max(self.max_error, float(diff.abs().max()))
            a, b = left.norm(dim=-1), right.norm(dim=-1)
            denominator = a * b
            cosine = (left * right).sum(-1) / denominator.clamp_min(torch.finfo(torch.float32).tiny)
            cosine[(a == 0) & (b == 0)] = 1.
            self.cosines.append(cosine.clamp(-1, 1))

    def result(self):
        cosine = torch.cat(self.cosines)
        # A zero reference has relative error zero only when the candidate is also zero.
        nmse = normalized_mse(self.error2, self.reference2)
        relative = math.sqrt(nmse) if nmse is not None else None
        result = {"tokens": len(cosine), "mean_cosine": float(cosine.mean()),
                  "minimum_cosine": float(cosine.min()), "first_percentile_cosine": float(cosine.quantile(.01)),
                  "relative_l2": relative, "normalized_mse": nmse,
                  "mse_normalization": "shared_hf_rms_per_site",
                  "maximum_absolute_error": self.max_error,
                  "error_squared_sum": self.error2, "reference_squared_sum": self.reference2}
        result["passed"] = result["mean_cosine"] >= .9999 and result["minimum_cosine"] >= .9990 and relative is not None and relative <= .02
        return result


def backend(args):
    torch.set_num_threads(min(16, os.cpu_count() or 1))
    manifest = json.loads((args.work / "manifest.json").read_text())
    data = manifest["languages"][args.language]
    path = args.work / f"{args.language}.safetensors"
    if sha256(path) != data["token_sha256"]:
        raise ValueError("Frozen token file checksum mismatch")
    all_tokens = load_file(str(path))["token_ids"]
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
            work=args.work, output=output, language=language, tokens=1_000_000,
            start=0, batch_size=args.batch_size, hf_python=args.hf_python,
        )
        if (output / "hf.json").exists() and (output / "vllm.json").exists():
            for engine in ("hf", "vllm"):
                receipt = json.loads((output / f"{engine}.json").read_text())
                expected = {"language": language, "tokens": 1_000_000, "start_token": 0,
                            "batch_size": args.batch_size,
                            "manifest_sha256": sha256(args.work / "manifest.json")}
                if any(receipt.get(key) != value for key, value in expected.items()):
                    raise ValueError(f"Completed receipts in {output} differ from this request")
            report(run_args)
        else:
            run(run_args)


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
    complete.add_argument("--batch-size", type=int, choices=(1, 2, 4, 8, 16, 32, 64, 128), default=4)
    complete.add_argument("--hf-python", help="Compatible Hugging Face Transformers Python, if separate")
    for command in ("run", "backend", "report"):
        child = sub.add_parser(command)
        child.add_argument("--output", type=Path, required=True)
        if command == "report":
            continue
        child.add_argument("--work", type=Path, required=True)
        child.add_argument("--language", choices=DATA, required=True)
        child.add_argument("--tokens", type=int, default=1_000_000)
        child.add_argument("--start", type=int, default=0)
        child.add_argument("--batch-size", type=int, choices=(1, 2, 4, 8, 16, 32, 64, 128), default=4)
        if command == "backend":
            child.add_argument("--backend", choices=("hf", "vllm"), required=True)
        else:
            child.add_argument("--hf-python", help="Existing compatible HF environment, if different from vLLM")
    args = parser.parse_args()
    {"prepare": prepare, "backend": backend, "run": run, "report": report, "all": run_all}[args.command](args)


if __name__ == "__main__":
    main()
