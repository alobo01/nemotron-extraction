"""Counterfactual expert-selection diagnostics; associations are labelled separately."""

from __future__ import annotations

import torch


def selection_summary(reference_ids, candidate_ids, reference_scores, candidate_scores):
    """Report set changes and the top-k stability bound for corrected router scores."""
    a, b = reference_ids.long(), candidate_ids.long()
    if a.ndim != 2 or a.shape != b.shape or reference_scores.shape != candidate_scores.shape:
        raise ValueError("Incompatible route or score shapes")
    n, k = a.shape
    if reference_scores.ndim != 2 or reference_scores.shape[0] != n or k >= reference_scores.shape[1]:
        raise ValueError("Expected matching rows and at least one excluded expert")
    for ids in (a, b):
        if bool(((ids < 0) | (ids >= reference_scores.shape[1])).any()):
            raise ValueError("Invalid expert ID")
        if k > 1 and bool((ids.sort(-1).values.diff(dim=-1) == 0).any()):
            raise ValueError("Duplicate expert ID")
    replacements = k - (a[:, :, None] == b[:, None, :]).any(-1).sum(-1)
    changed = replacements > 0
    sorted_scores = reference_scores.double().sort(-1, descending=True).values
    gaps = sorted_scores[:, k - 1] - sorted_scores[:, k]
    perturbation = (candidate_scores.double() - reference_scores.double()).abs().amax(-1)
    # For unconstrained top-k, gap > 2*max perturbation guarantees set stability.
    bound = gaps <= 2 * perturbation
    changed_gaps = gaps[changed]
    return {
        "tokens": n,
        "changed_rows": int(changed.sum()),
        "replacements_histogram": {str(i): int((replacements == i).sum()) for i in range(k + 1)},
        "exact_cutoff_ties": int((gaps == 0).sum()),
        "changed_gap_median": float(changed_gaps.median()) if changed_gaps.numel() else None,
        "changed_rows_satisfying_stability_bound": int((changed & bound).sum()),
    }, {"changed": changed, "replacements": replacements, "cutoff_gap": gaps, "score_perturbation": perturbation}


def counterfactual_summary(reference, candidate, forced_reference_routes, changed, *, thresholds=(0.99, 0.999)):
    """Hold candidate input/math fixed; quantify replacing its expert set.

    Routing weights must be recomputed from candidate logits for both candidate
    and forced outputs. Squared-error reduction includes interactions and may
    be negative; it is not an additive causal fraction of the original run.
    """
    if reference.ndim != 2 or reference.shape != candidate.shape or reference.shape != forced_reference_routes.shape:
        raise ValueError("Expected three matching token-by-hidden outputs")
    if changed.shape != (len(reference),):
        raise ValueError("Expected one route-change flag per token")
    h, v, c = (x.double() for x in (reference, candidate, forced_reference_routes))
    if not all(bool(torch.isfinite(x).all()) for x in (h, v, c)):
        raise ValueError("Nonfinite counterfactual output")

    def cosine(x):
        hn, xn = h.norm(dim=-1), x.norm(dim=-1)
        result = (h * x).sum(-1) / (hn * xn).clamp_min(torch.finfo(torch.float64).tiny)
        result[(hn == 0) & (xn == 0)] = 1
        return result.clamp(-1, 1)

    before, after = (v - h).square().sum(-1), (c - h).square().sum(-1)
    total, residual = float(before.sum()), float(after.sum())
    base_cos, forced_cos = cosine(v), cosine(c)
    reference2 = float(h.square().sum())
    result = {
        "tokens": len(h),
        "error_squared_before": total,
        "error_squared_after": residual,
        "relative_l2_before": (total / reference2) ** 0.5 if reference2 else None,
        "relative_l2_after": (residual / reference2) ** 0.5 if reference2 else None,
        "squared_error_reduction_fraction": 1 - residual / total if total else None,
        "error_energy_on_changed_rows_fraction": float(before[changed].sum()) / total if total else None,
        "bad_rows": {},
    }
    for threshold in thresholds:
        bad, fixed_bad = base_cos < threshold, forced_cos < threshold
        result["bad_rows"][str(threshold)] = {
            "before": int(bad.sum()),
            "after": int(fixed_bad.sum()),
            "with_route_changes": int((bad & changed).sum()),
            "recovered": int((bad & ~fixed_bad).sum()),
            "new": int((~bad & fixed_bad).sum()),
        }
    return result, {
        "cosine_before": base_cos,
        "cosine_after": forced_cos,
        "row_error_squared_before": before,
        "row_error_squared_after": after,
    }
