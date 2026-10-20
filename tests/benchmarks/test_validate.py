import argparse
import json

import pytest
import torch
from safetensors.torch import save_file

import validate
from benchlib.common import batches, digest, write_json


def test_exact_million_tokens_and_short_tail():
    tokens = torch.arange(1_000_000)
    chunks = list(batches(tokens, 4, 2048))
    assert sum(batch.numel() for batch in chunks) == 1_000_000
    assert chunks[-1].shape == (1, 576)
    assert torch.equal(torch.cat([batch.flatten() for batch in chunks]), tokens)


def test_all_languages_prepare_once_and_share_inputs(tmp_path, monkeypatch):
    args = argparse.Namespace(work=tmp_path / "tokens", output=tmp_path / "results",
                              revision=validate.MODEL_REVISION, batch_size=8, hf_python=None)
    stages, runs = [], []

    def prepare(value):
        stages.append(value.work)
        write_json(value.work / "manifest.json", {
            "model": validate.MODEL, "model_revision": value.revision})

    monkeypatch.setattr(validate, "prepare", prepare)
    monkeypatch.setattr(validate, "run", lambda value: runs.append(vars(value)))
    validate.run_all(args)
    assert stages == [args.work]
    assert [run["language"] for run in runs] == ["en", "de", "es"]
    assert all(run["work"] == args.work and run["tokens"] == 1_000_000
               and run["start"] == 0 and run["batch_size"] == 8 for run in runs)
    validate.run_all(args)
    assert stages == [args.work]  # Cached inputs must not be re-tokenised.


def test_all_languages_reject_partial_receipts_before_staging(tmp_path, monkeypatch):
    args = argparse.Namespace(work=tmp_path / "tokens", output=tmp_path / "results")
    write_json(args.output / "de/hf.json", {})
    monkeypatch.setattr(validate, "prepare", lambda _: pytest.fail("Must not overwrite partial results"))
    with pytest.raises(ValueError, match="Incomplete"):
        validate.run_all(args)


def test_all_languages_reject_stale_revision_and_completed_counts(tmp_path, monkeypatch):
    args = argparse.Namespace(work=tmp_path / "tokens", output=tmp_path / "results",
                              revision=validate.MODEL_REVISION, batch_size=4, hf_python=None)
    manifest = args.work / "manifest.json"
    write_json(manifest, {"model": validate.MODEL, "model_revision": "other"})
    monkeypatch.setattr(validate, "run", lambda _: pytest.fail("Must not run with mismatched inputs"))
    with pytest.raises(ValueError, match="revision"):
        validate.run_all(args)
    write_json(manifest, {"model": validate.MODEL, "model_revision": args.revision})
    record = {"language": "en", "tokens": 16_384, "start_token": 0, "batch_size": 4,
              "manifest_sha256": digest(manifest)}
    for engine in ("hf", "vllm"):
        write_json(args.output / f"en/{engine}.json", record)
    with pytest.raises(ValueError, match="differ"):
        validate.run_all(args)


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
    validate.aggregate_ranges([last, first], output, expected=2624)
    result = json.loads((output / "results.json").read_text())
    assert result["hf"]["tokens"] == 2624
    assert result["hf"]["tokens_per_second"] == 656
    assert result["vllm"]["agreement"]["L0_resid_post"]["relative_l2"] == pytest.approx(.00001 ** .5)
    assert result["passed"]
    with pytest.raises(ValueError, match="contiguous"):
        validate.aggregate_ranges([last], tmp_path / "missing", expected=576)
    with pytest.raises(ValueError, match="exactly"):
        validate.aggregate_ranges([first], tmp_path / "short")


def test_long_runs_split_into_contiguous_bounded_ranges(tmp_path):
    for language in ("en", "de", "es"):
        root = tmp_path / f"{language}-0"
        root.mkdir()
        values = {"tokens": 16384, "batch_size": 4, "startup_seconds": 60,
                  "extraction_seconds": 60, "persistence_seconds": 5,
                  "comparison_seconds": 10, "wall_seconds": 140}
        for backend in ("hf", "vllm"):
            (root / f"{backend}.json").write_text(json.dumps(values))
    jobs = validate.plan_ranges(tmp_path, 4)
    for language in ("en", "de", "es"):
        selected = [job for job in jobs if job["language"] == language]
        assert len(selected) > 1
        assert sum(job["tokens"] for job in selected) == 1_000_000
        end = 0
        for job in selected:
            assert job["start"] == end and end % 2048 == 0
            assert job["minutes"] <= 60 and job["minutes"] % 5 == 0
            end += job["tokens"]
    with pytest.raises(ValueError, match="batch size"):
        validate.plan_ranges(tmp_path, 2)


def test_all_languages_budget_for_cold_start(tmp_path):
    for language in ("en", "de", "es"):
        root = tmp_path / f"{language}-0"
        root.mkdir()
        for backend in ("hf", "vllm"):
            startup = (600 if backend == "hf" else 60) if language == "en" else 1
            value = {"tokens": 16384, "batch_size": 4, "startup_seconds": startup,
                     "extraction_seconds": 1, "persistence_seconds": 1,
                     "comparison_seconds": 1, "wall_seconds": startup + 2}
            (root / f"{backend}.json").write_text(json.dumps(value))
    jobs = validate.plan_ranges(tmp_path, 4)
    assert len(jobs) == 3
    assert {job["minutes"] for job in jobs} == {25}
