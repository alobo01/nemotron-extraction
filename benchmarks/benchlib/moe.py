"""Transparent PyTorch equations for Nemotron's non-gated mixture of experts.

They expose routing discontinuities and replay an unquantized expert layer.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .stats import same_expert_sets


def manual_sigmoid_router(hidden_states, gate_weight, correction_bias, top_k, *, gate_bias=None, routed_scale=1.0):
    """PyTorch equation for the checkpoint's one-group sigmoid top-k router.

    The correction bias changes expert choice only. Weights are gathered from
    the uncorrected sigmoid scores, normalized, then scaled.
    """
    x = hidden_states.float()
    logits = F.linear(x, gate_weight.float(), None if gate_bias is None else gate_bias.float())
    scores = torch.sigmoid(logits)
    ids = (scores + correction_bias.float()).topk(int(top_k), dim=-1, sorted=False).indices
    selected = scores.gather(-1, ids)
    weights = selected / selected.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)
    return {
        "logits": logits,
        "scores": scores,
        "ids": ids,
        "weights": weights * float(routed_scale),
        "normalized_weights": weights,
    }


def routing_report(logits, correction_bias, selected_ids):
    """Summarize observed routing without assuming how grouped top-k is encoded.

    ``selected_ids`` come from the backend's actual selector. Margins are measured
    against the global corrected sigmoid score; they are diagnostic only when
    grouped routing restricts eligibility.
    """
    corrected = torch.sigmoid(logits.float()) + correction_bias.float()
    ids = selected_ids.long()
    if ids.ndim != 2 or logits.shape[0] != ids.shape[0]:
        raise ValueError("Expected [tokens, experts] logits and matching [tokens, top_k] routes")
    if bool(((ids < 0) | (ids >= logits.shape[-1])).any()):
        raise ValueError("Selected expert ID is outside the router output range")

    rows = torch.arange(ids.shape[0], device=ids.device)[:, None]
    excluded = torch.ones_like(corrected, dtype=torch.bool)
    excluded.scatter_(1, ids, False)
    worst_selected = corrected[rows, ids].min(dim=1).values
    best_excluded = corrected.masked_fill(~excluded, -torch.inf).max(dim=1).values
    unconstrained = corrected.topk(ids.shape[1], dim=-1).indices
    return {
        "global_sixth_seventh_margin": (worst_selected - best_excluded).detach(),
        "unconstrained_set_agreement": same_expert_sets(ids, unconstrained).detach(),
    }


def manual_nemotron_moe(
    hidden_states,
    selected_ids,
    selected_weights,
    up_weight,
    down_weight,
    *,
    shared_up_weight=None,
    shared_down_weight=None,
    routed_scale=1.0,
    shared_up_bias=None,
    shared_down_bias=None,
    output_dtype=None,
):
    """Run the transparent ReLU-squared Nemotron-H expert equations.

    Weight layouts are ``[experts, intermediate, hidden]`` and
    ``[experts, hidden, intermediate]``. Per-expert matrix multiplies retain
    the checkpoint's input/weight dtype; weighted expert outputs accumulate
    in float32, matching the reference operation's stated accumulation.
    Scaling is applied to selected routing weights before accumulation.
    """
    x = hidden_states
    ids, weights = selected_ids.long(), selected_weights.float()
    if x.ndim != 2 or ids.ndim != 2 or weights.shape != ids.shape or ids.shape[0] != x.shape[0]:
        raise ValueError("Expected flattened tokens and matching [tokens, top_k] routes")
    if up_weight.ndim != 3 or down_weight.ndim != 3:
        raise ValueError("Expected expert weights with layouts [E,I,H] and [E,H,I]")
    if up_weight.shape[0] != down_weight.shape[0] or up_weight.shape[2] != x.shape[1]:
        raise ValueError("Expert weight dimensions do not match input")
    if down_weight.shape[1] != x.shape[1] or up_weight.shape[1] != down_weight.shape[2]:
        raise ValueError("Up/down expert dimensions do not match")
    if bool(((ids < 0) | (ids >= up_weight.shape[0])).any()):
        raise ValueError("Selected expert ID is outside the expert weight range")

    dtype = output_dtype or x.dtype
    routed = torch.zeros((x.shape[0], x.shape[1]), device=x.device, dtype=torch.float32)
    # The frozen Transformers source iterates expert IDs, not top-k slots.
    # Preserve this addition order and the per-expert GEMM row grouping.
    for expert in range(up_weight.shape[0]):
        rows, slots = torch.where(ids == expert)
        if not rows.numel():
            continue
        up = F.linear(x.index_select(0, rows), up_weight[expert])
        out = F.linear(F.relu(up).square(), down_weight[expert])
        routed.index_add_(0, rows, out * (weights[rows, slots] * float(routed_scale))[:, None])

    routed_cast = routed.to(dtype)
    shared = None
    if (shared_up_weight is None) != (shared_down_weight is None):
        raise ValueError("Shared up and down weights must be supplied together")
    if shared_up_weight is not None:
        shared_hidden = F.linear(x, shared_up_weight, shared_up_bias)
        shared = F.linear(F.relu(shared_hidden).square(), shared_down_weight, shared_down_bias)
    output = routed_cast if shared is None else routed_cast + shared.to(dtype)
    return {"output": output, "routed_cast": routed_cast}


def decompose_four_outputs(
    reference_on_reference, reference_on_candidate, candidate_on_reference, candidate_on_candidate
):
    """Return additive input/implementation effects from four observed outputs."""
    # Widen before subtracting: BF16 deltas would otherwise round separately
    # and obscure the algebraic decomposition being tested.
    hh, hv, vh, vv = (
        value.float()
        for value in (reference_on_reference, reference_on_candidate, candidate_on_reference, candidate_on_candidate)
    )
    return {
        "input_effect": hv - hh,
        "implementation_effect_on_candidate_input": vv - hv,
        "implementation_effect_on_reference_input": vh - hh,
        "interaction_effect": (vv - vh) - (hv - hh),
    }
