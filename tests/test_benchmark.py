import importlib.util
import argparse
from pathlib import Path

import pytest
import torch

spec = importlib.util.spec_from_file_location("benchmark", Path(__file__).parents[1] / "benchmarks/validate.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)



def test_exact_million_tokens_and_short_tail():
    tokens = torch.arange(1_000_000)
    batches = list(benchmark.batches(tokens, 4, 2048))
    assert sum(batch.numel() for batch in batches) == 1_000_000
    assert batches[-1].shape == (1, 576)
    assert torch.equal(torch.cat([batch.flatten() for batch in batches]), tokens)


def test_comparison_cache_size_is_automatic_unless_explicit(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "benchmarks"))
    comparison_spec = importlib.util.spec_from_file_location(
        "comparison", Path(__file__).parents[1] / "benchmarks/compare_repositories.py"
    )
    comparison = importlib.util.module_from_spec(comparison_spec)
    comparison_spec.loader.exec_module(comparison)
    monkeypatch.delenv("NH_KV_CACHE_BYTES", raising=False)
    assert comparison.configured_kv_cache_bytes() is None
    monkeypatch.setenv("NH_KV_CACHE_BYTES", "4096")
    assert comparison.configured_kv_cache_bytes() == 4096
    monkeypatch.delenv("NH_ENABLE_CHUNKED_PREFILL", raising=False)
    assert comparison.configured_chunked_prefill() is True
    monkeypatch.setenv("NH_ENABLE_CHUNKED_PREFILL", "false")
    assert comparison.configured_chunked_prefill() is False


def test_agreement_identical_zeros_and_known_error():
    metric = benchmark.Agreement()
    reference = torch.tensor([[1., 0.], [0., 0.]])
    metric.add(reference, reference.clone())
    result = metric.result()
    assert result["passed"] and result["minimum_cosine"] == 1
    drift = benchmark.Agreement()
    drift.add(reference[:1], torch.tensor([[1., 1.]]))
    assert drift.result()["relative_l2"] == 1
    assert drift.result()["normalized_mse"] == 1
    assert not drift.result()["passed"]
    zeros = benchmark.Agreement()
    zeros.add(reference[1:], torch.ones(1, 2))
    assert zeros.result()["relative_l2"] is None
    assert zeros.result()["normalized_mse"] is None
    assert result["normalized_mse"] == 0
    assert not zeros.result()["passed"]


def test_metrics_reject_shape_or_dtype_mismatch():
    with pytest.raises(ValueError, match="differ"):
        benchmark.Agreement().add(torch.zeros(1, 2), torch.zeros(2, 2))


def test_normalized_mse_uses_shared_reference_scale():
    reference = torch.tensor([[1., 2.], [3., 4.]])
    candidate = reference * 1.5
    metric = benchmark.Agreement()
    metric.add(reference, candidate)
    result = metric.result()
    # Independent unit normalisation would hide this magnitude error.
    assert result["mean_cosine"] == pytest.approx(1.)
    assert result["normalized_mse"] == pytest.approx(.25)
    rms = reference.double().square().mean().sqrt()
    expected = ((candidate.double() / rms - reference.double() / rms) ** 2).mean()
    assert result["normalized_mse"] == pytest.approx(float(expected))
    assert result["normalized_mse"] == pytest.approx(result["relative_l2"] ** 2)
    for scale in (.125, 128.):
        scaled = benchmark.Agreement()
        scaled.add(reference.repeat(3, 2) * scale, candidate.repeat(3, 2) * scale)
        assert scaled.result()["normalized_mse"] == pytest.approx(.25)


def test_normalized_mse_accumulates_before_normalization():
    metric = benchmark.Agreement()
    metric.add(torch.tensor([[1., 0.]]), torch.tensor([[2., 0.]]))
    metric.add(torch.tensor([[10., 0.]]), torch.tensor([[10., 0.]]))
    assert metric.result()["normalized_mse"] == pytest.approx(1 / 101)


def test_all_languages_prepare_once_and_share_inputs(tmp_path, monkeypatch):
    args = argparse.Namespace(work=tmp_path / "tokens", output=tmp_path / "results",
                              revision=benchmark.MODEL_REVISION, batch_size=8, hf_python=None)
    stages, runs = [], []

    def prepare(value):
        stages.append(value.work)
        benchmark.write_json(value.work / "manifest.json", {
            "model": benchmark.MODEL, "model_revision": value.revision})

    monkeypatch.setattr(benchmark, "prepare", prepare)
    monkeypatch.setattr(benchmark, "run", lambda value: runs.append(vars(value)))
    benchmark.run_all(args)
    assert stages == [args.work]
    assert [run["language"] for run in runs] == ["en", "de", "es"]
    assert all(run["work"] == args.work and run["tokens"] == 1_000_000
               and run["start"] == 0 and run["batch_size"] == 8 for run in runs)
    benchmark.run_all(args)
    assert stages == [args.work]  # Cached inputs must not be re-tokenised.


def test_all_languages_reject_partial_receipts_before_staging(tmp_path, monkeypatch):
    args = argparse.Namespace(work=tmp_path / "tokens", output=tmp_path / "results")
    benchmark.write_json(args.output / "de/hf.json", {})
    monkeypatch.setattr(benchmark, "prepare", lambda _: pytest.fail("Must not overwrite partial results"))
    with pytest.raises(ValueError, match="Incomplete"):
        benchmark.run_all(args)


def test_all_languages_reject_stale_revision_and_completed_counts(tmp_path, monkeypatch):
    args = argparse.Namespace(work=tmp_path / "tokens", output=tmp_path / "results",
                              revision=benchmark.MODEL_REVISION, batch_size=4, hf_python=None)
    manifest = args.work / "manifest.json"
    benchmark.write_json(manifest, {"model": benchmark.MODEL, "model_revision": "other"})
    monkeypatch.setattr(benchmark, "run", lambda _: pytest.fail("Must not run with mismatched inputs"))
    with pytest.raises(ValueError, match="revision"):
        benchmark.run_all(args)
    benchmark.write_json(manifest, {"model": benchmark.MODEL, "model_revision": args.revision})
    record = {"language": "en", "tokens": 16_384, "start_token": 0, "batch_size": 4,
              "manifest_sha256": benchmark.sha256(manifest)}
    for engine in ("hf", "vllm"):
        benchmark.write_json(args.output / f"en/{engine}.json", record)
    with pytest.raises(ValueError, match="differ"):
        benchmark.run_all(args)
