"""Known counterfactuals distinguish route association from error removal."""

import pytest
import torch
from benchlib.attribution import counterfactual_summary, selection_summary


def test_selection_ignores_slot_order_and_measures_one_expert_swap():
    scores = torch.tensor([[0.9, 0.8, 0.79], [0.9, 0.8, 0.79]])
    changed_scores = scores.clone()
    changed_scores[1, 2] += 0.02
    result, rows = selection_summary(
        torch.tensor([[0, 1], [0, 1]]), torch.tensor([[1, 0], [2, 0]]), scores, changed_scores
    )
    assert result["changed_rows"] == 1
    assert result["replacements_histogram"] == {"0": 1, "1": 1, "2": 0}
    assert result["changed_rows_satisfying_stability_bound"] == 1
    assert rows["changed"].tolist() == [False, True]
    with pytest.raises(ValueError, match="Duplicate"):
        selection_summary(torch.tensor([[0, 0], [0, 1]]), torch.tensor([[0, 1], [0, 1]]), scores, scores)


def test_counterfactual_recovers_route_error_but_preserves_continuous_error():
    h = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    v = torch.tensor([[0.0, 1.0], [2.0, 0.0]])
    c = torch.tensor([[1.0, 0.0], [2.0, 0.0]])
    result, _ = counterfactual_summary(h, v, c, torch.tensor([True, False]))
    assert result["bad_rows"]["0.99"] == {"before": 1, "after": 0, "with_route_changes": 1, "recovered": 1, "new": 0}
    assert result["squared_error_reduction_fraction"] == pytest.approx(2 / 3)
    assert result["error_squared_after"] == 1


def test_interaction_can_increase_error_and_zero_energy_is_undefined():
    h, v, c = (torch.tensor([[x, 0.0]]) for x in (1.0, 2.0, 3.0))
    result, _ = counterfactual_summary(h, v, c, torch.tensor([True]))
    assert result["squared_error_reduction_fraction"] == -3
    result, _ = counterfactual_summary(h, h, h, torch.tensor([False]))
    assert result["squared_error_reduction_fraction"] is None


def test_identical_scores_can_select_different_valid_experts_at_a_tie():
    scores = torch.tensor([[0.9, 0.8, 0.8]])
    result, _ = selection_summary(torch.tensor([[0, 1]]), torch.tensor([[0, 2]]), scores, scores)
    assert result["changed_rows"] == result["exact_cutoff_ties"] == 1
    assert result["changed_rows_satisfying_stability_bound"] == 1
