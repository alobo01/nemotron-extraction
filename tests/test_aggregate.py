import importlib.util
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

spec = importlib.util.spec_from_file_location("aggregate", Path(__file__).parents[1] / "benchmarks/aggregate.py")
aggregate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(aggregate)


def receipt(path, start, count, cosine, error, norm):
    path.mkdir()
    common = {"language": "en", "tokens": count, "start_token": start, "batch_size": 4,
              "sequence_length": 2048, "manifest_sha256": "fixed", "script_sha256": "fixed",
              "topology": {"L0_resid_post": {"width": 2}}, "software": {}, "gpu": "fixture",
              "source": {"model": "fixture", "dtype": "bfloat16"},
              "model_load_seconds": 1, "hook_install_seconds": 0, "startup_seconds": 1,
              "warmup_seconds": .1, "inference_seconds": 2, "capture_copy_seconds": .1,
              "worker_write_seconds": .1, "worker_collection_seconds": .1, "driver_read_seconds": .1,
              "gpu_memory_utilization": None, "kv_cache_memory_bytes": None,
              "cuda_graphs_enabled": False, "extraction_seconds": 2, "comparison_seconds": 3,
              "persistence_seconds": 4, "wall_seconds": 10, "tokens_per_second": count / 2,
              "peak_gpu_memory_bytes": 123}
    for backend in ("hf", "vllm"):
        value = {**common, "backend": backend}
        if backend == "vllm":
            value["gpu_memory_utilization"] = .96
            value.update(passed=True, agreement={"L0_resid_post": {"tokens": count,
                "mean_cosine": cosine, "minimum_cosine": cosine, "first_percentile_cosine": cosine,
                "relative_l2": (error / norm) ** .5, "maximum_absolute_error": .001,
                "error_squared_sum": error, "reference_squared_sum": norm, "passed": True}})
        (path / f"{backend}.json").write_text(json.dumps(value))
    save_file({"L0_resid_post": torch.full((count,), cosine)}, str(path / "cosines.safetensors"))


def test_aggregation_exact_coverage_and_global_relative_error(tmp_path):
    first, last = tmp_path / "first", tmp_path / "last"
    receipt(first, 0, 2048, 1., .0001, 10)
    receipt(last, 2048, 576, 1., .0002, 20)
    output = tmp_path / "combined"
    aggregate.aggregate([last, first], output, expected=2624)
    result = json.loads((output / "results.json").read_text())
    assert result["hf"]["tokens"] == 2624
    assert result["hf"]["tokens_per_second"] == 656
    assert result["vllm"]["agreement"]["L0_resid_post"]["relative_l2"] == pytest.approx(.00001 ** .5)
    assert result["passed"]
    with pytest.raises(ValueError, match="contiguous"):
        aggregate.aggregate([last], tmp_path / "missing", expected=576)
    with pytest.raises(ValueError, match="exactly"):
        aggregate.aggregate([first], tmp_path / "short")
