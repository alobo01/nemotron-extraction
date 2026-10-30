import json

import pytest
import torch
from benchlib.stats import Agreement, TokenDiagnostics
from safetensors.torch import load_file


def test_agreement_identical_zeros_and_known_error():
    metric = Agreement()
    reference = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    metric.add(reference, reference.clone())
    result = metric.result()
    assert result["passed"] and result["minimum_cosine"] == 1
    drift = Agreement()
    drift.add(reference[:1], torch.tensor([[1.0, 1.0]]))
    assert drift.result()["relative_l2"] == 1
    assert drift.result()["normalized_mse"] == 1
    assert not drift.result()["passed"]
    zeros = Agreement()
    zeros.add(reference[1:], torch.ones(1, 2))
    assert zeros.result()["relative_l2"] is None
    assert zeros.result()["normalized_mse"] is None
    assert result["normalized_mse"] == 0
    assert not zeros.result()["passed"]


def test_metrics_reject_shape_or_dtype_mismatch():
    with pytest.raises(ValueError, match="differ"):
        Agreement().add(torch.zeros(1, 2), torch.zeros(2, 2))


def test_normalized_mse_uses_shared_reference_scale():
    reference = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    candidate = reference * 1.5
    metric = Agreement()
    metric.add(reference, candidate)
    result = metric.result()
    # Independent unit normalisation would hide this magnitude error.
    assert result["mean_cosine"] == pytest.approx(1.0)
    assert result["normalized_mse"] == pytest.approx(0.25)
    rms = reference.double().square().mean().sqrt()
    expected = ((candidate.double() / rms - reference.double() / rms) ** 2).mean()
    assert result["normalized_mse"] == pytest.approx(float(expected))
    assert result["normalized_mse"] == pytest.approx(result["relative_l2"] ** 2)
    for scale in (0.125, 128.0):
        scaled = Agreement()
        scaled.add(reference.repeat(3, 2) * scale, candidate.repeat(3, 2) * scale)
        assert scaled.result()["normalized_mse"] == pytest.approx(0.25)


def test_worst_one_percent_tails_cover_the_worst_tokens():
    reference = torch.ones(200, 2)
    candidate = reference.clone()
    candidate[7] = torch.tensor([1.0, 3.0])
    candidate[8] = torch.tensor([-1.0, 1.0])
    metric = Agreement()
    metric.add(reference, candidate)
    result = metric.result()
    # Two tokens are the worst 1% of 200: their errors are 4 and 4, against 2 reference energy per token.
    assert result["worst_one_percent_normalized_mse"] == pytest.approx(2.0)
    assert result["worst_one_percent_mean_cosine"] == pytest.approx((0.0 + 0.8944272) / 2, abs=1e-6)
    merged = Agreement()
    merged.merge_result(result, torch.cat(metric.cosines))
    assert merged.result()["worst_one_percent_normalized_mse"] is None


def test_normalized_mse_accumulates_before_normalization():
    metric = Agreement()
    metric.add(torch.tensor([[1.0, 0.0]]), torch.tensor([[2.0, 0.0]]))
    metric.add(torch.tensor([[10.0, 0.0]]), torch.tensor([[10.0, 0.0]]))
    assert metric.result()["normalized_mse"] == pytest.approx(1 / 101)


def ids(n, offset=0):
    return [
        dict(
            row_id=f"row-{i:04d}",
            sequence_id="short" if i < 37 else "long",
            position=i if i < 37 else i - 37,
            token_id=i,
        )
        for i in range(offset, offset + n)
    ]


def test_zero_vectors_are_explicit_and_gates_unchanged():
    h = torch.tensor([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
    v = torch.tensor([[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0]])
    d = TokenDiagnostics()
    d.add(h, v, ids(3))
    r = d.result()
    assert r["both_zero_rows"] == 1 and r["one_zero_rows"] == 1
    assert r["minimum_cosine"] == -1 and not r["passed"]
    assert r["normalized_mse"] == 5 and r["relative_l2"] == pytest.approx(5**0.5)
    assert r["worst_cosine"][1]["relative_error_undefined"]
    json.dumps(r, allow_nan=False)


def test_shard_merge_ranking_and_error_budget_match_full_arrays(tmp_path):
    gen = torch.Generator().manual_seed(8)
    h = torch.randn(75, 9, generator=gen)
    v = h.clone()
    v[55] *= -1
    v[72] += 2
    whole = TokenDiagnostics()
    whole.add(h, v, ids(75))
    merged = TokenDiagnostics()
    for a, b in ((0, 37), (37, 68), (68, 75)):
        piece = TokenDiagnostics()
        piece.add(h[a:b], v[a:b], ids(b - a, a))
        merged.merge(piece)
    assert merged.result() == whole.result()
    assert merged.result()["worst_cosine"][0]["position"] == 18
    assert merged.result()["worst_error"][0]["row_id"] in ("row-0055", "row-0072")
    assert len(merged.samples) <= 40
    merged.save_samples(tmp_path / "worst.safetensors")
    assert load_file(str(tmp_path / "worst.safetensors"))
    with pytest.raises(ValueError, match="Duplicate"):
        merged.merge(whole)


def test_normalization_retains_scale_error_and_rejects_bad_rows():
    h = torch.ones(40, 3, dtype=torch.bfloat16)
    v = h * 2
    d = TokenDiagnostics()
    d.add(h, v, ids(40))
    r = d.result()
    assert r["normalized_mse"] == 1 and not r["passed"]
    assert r["mean_cosine"] == pytest.approx(1, abs=1e-6)
    with pytest.raises(ValueError, match="Nonfinite"):
        TokenDiagnostics().add(h.float(), h.float() * float("nan"), ids(40))
    with pytest.raises(ValueError, match="identity"):
        TokenDiagnostics().add(h, v, ids(39))


def test_worst_one_percent_reports_extremes_and_merges_without_changing_ranking():
    a = torch.ones(101, 2)
    b = a.clone()
    b[7] *= -1
    b[100] *= 3
    identities = [dict(row_id=str(i), sequence_id="anchor", position=i, token_id=i) for i in range(101)]
    whole, left, right = TokenDiagnostics(), TokenDiagnostics(), TokenDiagnostics()
    whole.add(a, b, identities)
    left.add(a[:50], b[:50], identities[:50])
    right.add(a[50:], b[50:], identities[50:])
    observed = whole.result()
    assert observed == left.merge(right).result()
    tail = observed["worst_one_percent"]
    assert tail["count"] == 2
    assert {r["position"] for r in tail["rows_by_squared_error"]} == {7, 100}
    assert tail["error_energy_share"] == 1
    assert observed["maximum_cosine_disagreement"] == pytest.approx(2)
    assert observed["maximum_absolute_error"] == 2
    errors = (b - a).double().square().sum(-1) / a.double().square().sum(-1).mean()
    assert observed["normalized_mse_std"] == pytest.approx(float(errors.std(correction=0)))
