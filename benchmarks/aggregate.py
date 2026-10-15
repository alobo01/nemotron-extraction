# /// script
# requires-python = ">=3.10"
# dependencies = ["nemotron-extraction==1.0.0"]
# ///
"""Aggregate aligned range receipts into exactly 1M validated tokens/language."""
import argparse
import importlib.util
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

spec = importlib.util.spec_from_file_location("validate", Path(__file__).with_name("validate.py"))
validate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validate)


def aggregate(paths, output, expected=1_000_000):
    receipts = []
    for path in paths:
        hf = json.loads((path / "hf.json").read_text())
        vllm = json.loads((path / "vllm.json").read_text())
        validate.report(argparse.Namespace(output=path))
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
            target = combined.setdefault(key, validate.Agreement())
            target.cosines.append(cosines[key])
            target.error2 += metric["error_squared_sum"]
            target.reference2 += metric["reference_squared_sum"]
            target.max_error = max(target.max_error, metric["maximum_absolute_error"])
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
        validate.write_json(output / f"{engine}.json", result)
    validate.report(argparse.Namespace(output=output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    aggregate(args.outputs, args.output)
