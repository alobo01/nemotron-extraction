"""Equation-level diagnostics for Nemotron's non-gated mixture of experts.

These helpers are intentionally separate from extraction and inference paths.
They expose routing discontinuities and provide a transparent PyTorch replay
for an unquantized, non-gated expert layer.
"""
from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F


def manual_sigmoid_router(hidden_states, gate_weight, correction_bias, top_k,
                          *, gate_bias=None, routed_scale=1.0):
    """PyTorch equation for the checkpoint's one-group sigmoid top-k router.

    The correction bias changes expert choice only. Weights are gathered from
    the uncorrected sigmoid scores, normalized, then scaled.
    """
    x=hidden_states.float()
    logits=F.linear(x,gate_weight.float(),None if gate_bias is None else gate_bias.float())
    scores=torch.sigmoid(logits)
    ids=(scores+correction_bias.float()).topk(int(top_k),dim=-1,sorted=False).indices
    selected=scores.gather(-1,ids)
    weights=selected/selected.sum(dim=-1,keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)
    return {'logits':logits,'scores':scores,'ids':ids,'weights':weights*float(routed_scale),
            'normalized_weights':weights}


def routing_report(logits, correction_bias, selected_ids, selected_weights):
    """Summarize observed routing without assuming how grouped top-k is encoded.

    ``selected_ids`` and ``selected_weights`` come from the backend's actual
    selector. Margins are measured against the global corrected sigmoid score;
    they are diagnostic only when grouped routing restricts eligibility.
    """
    scores = torch.sigmoid(logits.float())
    corrected = scores + correction_bias.float()
    ids = selected_ids.long()
    weights = selected_weights.float()
    if ids.ndim != 2 or weights.shape != ids.shape or logits.shape[0] != ids.shape[0]:
        raise ValueError("Expected [tokens, experts] logits and matching [tokens, top_k] routes")
    if bool(((ids < 0) | (ids >= logits.shape[-1])).any()):
        raise ValueError("Selected expert ID is outside the router output range")

    rows = torch.arange(ids.shape[0], device=ids.device)[:, None]
    selected_corrected = corrected[rows, ids]
    excluded = torch.ones_like(corrected, dtype=torch.bool)
    excluded.scatter_(1, ids, False)
    worst_selected = selected_corrected.min(dim=1).values
    best_excluded = corrected.masked_fill(~excluded, -torch.inf).max(dim=1).values
    margins = worst_selected - best_excluded
    k = ids.shape[1]
    unconstrained = corrected.topk(k, dim=-1).indices
    agree = torch.sort(ids, dim=-1).values == torch.sort(unconstrained, dim=-1).values
    chosen_scores = scores[rows, ids]
    normalized = chosen_scores / chosen_scores.sum(dim=1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)
    return {
        "router_logits": logits.detach(),
        "sigmoid_scores": scores.detach(),
        "corrected_scores": corrected.detach(),
        "selected_ids": ids.detach(),
        "selected_weights": weights.detach(),
        "selected_scores_before_normalization": chosen_scores.detach(),
        "normalized_sigmoid_weights": normalized.detach(),
        "global_sixth_seventh_margin": margins.detach(),
        "unconstrained_topk_ids": unconstrained.detach(),
        "unconstrained_set_agreement": agree.all(dim=1).detach(),
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
    up_bias=None,
    down_bias=None,
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
    per_slot = torch.zeros((x.shape[0], ids.shape[1], x.shape[1]), device=x.device, dtype=torch.float32)
    # The frozen Transformers source iterates expert IDs, not top-k slots.
    # Preserve this addition order and the per-expert GEMM row grouping.
    for expert in range(up_weight.shape[0]):
        rows, slots = torch.where(ids == expert)
        if not rows.numel():
            continue
        up = F.linear(x.index_select(0, rows), up_weight[expert],
                      None if up_bias is None else up_bias[expert])
        out = F.linear(F.relu(up).square(), down_weight[expert],
                       None if down_bias is None else down_bias[expert])
        contribution = out * (weights[rows, slots] * float(routed_scale))[:, None]
        routed.index_add_(0, rows, contribution)
        per_slot[rows, slots] = contribution

    routed_cast = routed.to(dtype)
    shared = None
    if (shared_up_weight is None) != (shared_down_weight is None):
        raise ValueError("Shared up and down weights must be supplied together")
    if shared_up_weight is not None:
        shared_hidden = F.linear(x, shared_up_weight, shared_up_bias)
        shared = F.linear(F.relu(shared_hidden).square(), shared_down_weight, shared_down_bias)
    output = routed_cast if shared is None else routed_cast + shared.to(dtype)
    return {
        "output": output,
        "routed_float32": routed,
        "routed_cast": routed_cast,
        "shared_output": shared,
        "weighted_slot_contributions": per_slot,
    }


def four_way_replay(reference_input, candidate_input, reference_op: Callable, candidate_op: Callable):
    """Evaluate both implementations on both inputs and separate effects.

    The exact decomposition is ``candidate(candidate_input)-reference(reference_input)``
    = input effect ``reference(candidate_input)-reference(reference_input)`` +
    implementation effect ``candidate(candidate_input)-reference(candidate_input)``.
    The crossed candidate-on-reference result is retained to expose interaction.
    """
    hh = reference_op(reference_input)
    hv = reference_op(candidate_input)
    vh = candidate_op(reference_input)
    vv = candidate_op(candidate_input)
    return decompose_four_outputs(hh, hv, vh, vv)


def decompose_four_outputs(reference_on_reference, reference_on_candidate,
                           candidate_on_reference, candidate_on_candidate):
    """Return additive input/implementation effects from four observed outputs."""
    # Widen before subtracting: BF16 deltas would otherwise round separately
    # and obscure the algebraic decomposition being tested.
    hh,hv,vh,vv=(value.float() for value in (reference_on_reference,
        reference_on_candidate,candidate_on_reference,candidate_on_candidate))
    return {
        "reference_on_reference_input": hh,
        "reference_on_candidate_input": hv,
        "candidate_on_reference_input": vh,
        "candidate_on_candidate_input": vv,
        "input_effect": hv - hh,
        "implementation_effect_on_candidate_input": vv - hv,
        "implementation_effect_on_reference_input": vh - hh,
        "interaction_effect": (vv - vh) - (hv - hh),
        "reconstruction_error": (vv - hh) - ((hv - hh) + (vv - hv)),
    }
