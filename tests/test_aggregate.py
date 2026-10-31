"""Group sums on the worker: span routing, chunked prompts and sums across forwards, on CPU."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from nemotron_extraction.aggregate import AggregateHookExtension, decode_result, group_weights


class SplitBlock(nn.Module):
    def forward(self, positions, hidden_states, residual):
        return hidden_states * 2, residual + hidden_states


class WorkerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = nn.Module()
        self.language_model.model = nn.Module()
        self.language_model.model.layers = nn.ModuleList([SplitBlock(), SplitBlock()])

    def forward(self, input_ids, positions):
        hidden = input_ids.float().unsqueeze(1).repeat(1, 3)
        residual = torch.ones_like(hidden)
        for block in self.language_model.model.layers:
            hidden, residual = block(positions, hidden, residual)
        return hidden


@pytest.fixture
def worker(monkeypatch):
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    instance = AggregateHookExtension()
    instance.model_runner = SimpleNamespace(model=WorkerModel())
    instance.ag_install([0, 1], groups=2)
    yield instance
    instance.ag_remove()
    assert not instance.model_runner.model._forward_hooks


def schedule(worker, entries):
    """One forward over (request ID, full prompt, computed offset, scheduled count) entries."""
    worker.model_runner.input_batch = SimpleNamespace(
        req_ids=[rid for rid, *_ in entries], num_computed_tokens_cpu=[computed for _, _, computed, _ in entries]
    )
    counts = torch.tensor([count for *_, count in entries])
    worker.model_runner.query_start_loc = SimpleNamespace(cpu=torch.cat([torch.zeros(1), counts.cumsum(0)]).long())
    # Positions past the prompt stand in for generated tokens.
    ids = [(prompt + [99] * count)[computed : computed + count] for _, prompt, computed, count in entries]
    flat = torch.tensor(sum(ids, []))
    worker.model_runner.model(flat, torch.arange(len(flat)))


def block_output(token, layer):
    """Layer 0 gives 2t + (1 + t) = 3t + 1 over three columns; layer 1 doubles hidden and adds it to the residual."""
    first_hidden, first_residual = 2 * token, 1 + token
    if layer == 0:
        return first_hidden + first_residual
    return 2 * first_hidden + first_residual + first_hidden


def test_group_weights_assign_rows_and_leave_padding_out():
    weights = group_weights([(0, 2, 1), (2, 1, 0)], rows=5, groups=2)
    assert weights.tolist() == [[0, 0, 1, 0, 0], [1, 1, 0, 0, 0]]


def test_sums_cover_prompt_tokens_only_across_chunks_and_requests(worker):
    worker.ag_begin({"a": (0, 3), "b": (1, 2), "c": (0, 1)})
    schedule(worker, [("a-01234567", [5, 6, 7], 0, 2), ("b-89abcdef", [8, 9], 0, 2)])
    schedule(worker, [("a-01234567", [5, 6, 7], 2, 1), ("c-deadbeef", [4], 0, 1)])
    sums, counts = decode_result(worker.ag_result())
    assert counts.tolist() == [4, 2]
    for slot in (0, 1):
        expected_zero = sum(block_output(t, slot) for t in (5, 6, 7, 4))
        expected_one = sum(block_output(t, slot) for t in (8, 9))
        assert sums[slot, 0].tolist() == [expected_zero] * 3
        assert sums[slot, 1].tolist() == [expected_one] * 3


def test_generated_tokens_are_not_counted(worker):
    worker.ag_begin({"a": (0, 1)})
    schedule(worker, [("a", [5], 0, 1)])
    schedule(worker, [("a", [5], 1, 1)])
    sums, counts = decode_result(worker.ag_result())
    assert counts.tolist() == [1, 0]
    assert sums[0, 0].tolist() == [block_output(5, 0)] * 3


def test_out_of_order_span_is_rejected(worker):
    worker.ag_begin({"a": (0, 3)})
    with pytest.raises(RuntimeError, match="out of order"):
        schedule(worker, [("a", [5, 6, 7], 1, 2)])


def test_unfinished_prompt_fails_the_result(worker):
    worker.ag_begin({"a": (0, 3)})
    schedule(worker, [("a", [5, 6, 7], 0, 2)])
    with pytest.raises(RuntimeError, match="Missing prompt activations"):
        worker.ag_result()


def test_duplicate_request_id_is_rejected(worker):
    worker.ag_begin({"a": (0, 1)})
    with pytest.raises(ValueError, match="already registered"):
        worker.ag_begin({"a": (1, 1)})


def test_reset_discards_a_warmup_pass(worker):
    worker.ag_begin({"w": (0, 2)})
    schedule(worker, [("w", [5, 6], 0, 2)])
    worker.ag_reset()
    worker.ag_begin({"a": (1, 1)})
    schedule(worker, [("a", [7], 0, 1)])
    sums, counts = decode_result(worker.ag_result())
    assert counts.tolist() == [0, 1]
    assert sums[0, 0].tolist() == [0.0] * 3
    assert sums[0, 1].tolist() == [block_output(7, 0)] * 3
