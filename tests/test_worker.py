"""Exercise worker protocol and row identity with a CPU CUDA-event stand-in."""

import time
from concurrent.futures import Future
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file
from torch import nn

from nemotron_extraction.transport import BufferPool
from nemotron_extraction.worker import ActivationHookExtension


class Event:
    def __init__(self, **kwargs):
        pass

    def record(self):
        pass

    def synchronize(self):
        pass

    def elapsed_time(self, other):
        return 1.0

    def wait_stream(self, stream):
        pass


class SplitBlock(nn.Module):
    def forward(self, positions, hidden_states, residual):
        return hidden_states * 2, residual + hidden_states


class WorkerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([SplitBlock()])

    def forward(self, input_ids, positions):
        hidden = input_ids.float().unsqueeze(1)
        return self.model.layers[0](positions, hidden, torch.ones_like(hidden))


@pytest.fixture
def worker(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.cuda, "Stream", Event)
    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    monkeypatch.setattr(torch.cuda, "current_stream", Event)
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda tensor, stream: None)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (80, 100))
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 123)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 123)
    original = BufferPool.tensor
    monkeypatch.setattr(
        BufferPool,
        "tensor",
        lambda self, slot, key, shape, dtype, **kw: original(self, slot, key, shape, dtype, pinned=False),
    )
    instance = ActivationHookExtension()
    instance.model_runner = SimpleNamespace(model=WorkerModel())
    instance.nh_install([{"layer": 0, "kind": "resid_pre"}, {"layer": 0, "kind": "resid_post"}], str(tmp_path))
    yield instance
    instance.nh_remove()
    assert not instance.model_runner.model._forward_hooks


def forward(worker, request_ids, rows, computed=None):
    worker.model_runner.input_batch = SimpleNamespace(
        req_ids=request_ids, token_ids_cpu=torch.tensor(rows), num_computed_tokens_cpu=computed or [0] * len(rows)
    )
    length = len(rows[0])
    worker.model_runner.query_start_loc = SimpleNamespace(cpu=torch.arange(len(rows) + 1) * length)
    worker.model_runner.model(torch.tensor(rows).flatten(), torch.arange(length).repeat(len(rows)))


def poll(worker):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        receipts = worker.nh_poll()
        if receipts:
            return receipts
        time.sleep(0.005)
    raise AssertionError("Writer did not finish")


def test_request_order_duplicates_and_internal_suffix(worker, tmp_path):
    rows = [[1, 2], [3, 4], [1, 2]]
    assert worker.nh_begin("wave", ["a", "b", "c"], rows, str(tmp_path / "shard.safetensors"))
    forward(worker, ["b-01234567", "c-89abcdef", "a-deadbeef"], [rows[1], rows[2], rows[0]])
    worker.nh_seal("wave")
    receipt = poll(worker)[0]
    result = load_file(receipt["path"])
    assert torch.equal(result["token_ids"], torch.tensor(rows))
    assert result["L0_resid_pre"].flatten().tolist() == [2, 3, 4, 5, 2, 3]
    assert result["L0_resid_post"].flatten().tolist() == [4, 7, 10, 13, 4, 7]
    assert receipt["actual_forward_batches"] == [3]
    assert receipt["peak_device_memory_bytes"] == 20
    assert receipt["total_device_memory_bytes"] == 100
    assert receipt["peak_device_memory_fraction"] == pytest.approx(0.2)
    assert sorted(worker.nh_pool.free) == [0, 1]


def test_poll_does_not_wait_and_propagates_write_failure(worker):
    future = Future()
    slot = worker.nh_pool.acquire()
    worker.nh_pending["failed"] = (slot, future)
    assert worker.nh_poll() == []
    future.set_exception(OSError("disk full"))
    with pytest.raises(OSError, match="disk full"):
        worker.nh_poll()
    assert slot in worker.nh_pool.free


def test_unknown_request_and_partial_prefill_are_rejected(worker, tmp_path):
    worker.nh_begin("wave", ["a"], [[1, 2]], str(tmp_path / "x.safetensors"))
    with pytest.raises(RuntimeError, match="Unknown request"):
        forward(worker, ["wrong"], [[1, 2]])
    with pytest.raises(RuntimeError, match="partial prefill"):
        forward(worker, ["a"], [[1, 2]], computed=[1])


def test_detailed_token_validation_is_opt_in_but_scheduler_checks_remain(worker, tmp_path):
    worker.nh_diagnostics = True
    worker.nh_begin("tokens", ["a"], [[1, 2]], str(tmp_path / "tokens.safetensors"))
    with pytest.raises(RuntimeError, match="token IDs differ"):
        forward(worker, ["a"], [[2, 1]])
    assert not worker.nh_active["covered"]


def test_detailed_token_validation_accepts_numpy_scheduler_tokens(worker, tmp_path):
    worker.nh_diagnostics = True
    worker.nh_begin("tokens", ["a"], [[1, 2]], str(tmp_path / "tokens.safetensors"))
    worker.model_runner.input_batch = SimpleNamespace(
        req_ids=["a"], token_ids_cpu=torch.tensor([[1, 2]]).numpy(), num_computed_tokens_cpu=[0]
    )
    worker.model_runner.query_start_loc = SimpleNamespace(cpu=torch.tensor([0, 2]))
    worker.model_runner.model(torch.tensor([1, 2]), torch.tensor([0, 1]))
    assert worker.nh_active["covered"] == {0}


def test_fast_path_does_not_read_per_token_cpu_content(worker, tmp_path):
    class SchedulerTokens:
        def __getitem__(self, key):
            raise AssertionError("Detailed token-content access on fast path")

    worker.nh_begin("fast", ["a"], [[1, 2]], str(tmp_path / "fast.safetensors"))
    worker.model_runner.input_batch = SimpleNamespace(
        req_ids=["a"], token_ids_cpu=SchedulerTokens(), num_computed_tokens_cpu=[0]
    )
    worker.model_runner.query_start_loc = SimpleNamespace(cpu=torch.tensor([0, 2]))
    worker.model_runner.model(torch.tensor([1, 2]), torch.tensor([0, 1]))
    worker.nh_seal("fast")
    r = poll(worker)[0]
    assert r["scheduler"] == [
        dict(request_id="a", sequence_index=0, start=0, stop=2, computed_tokens=0, prompt_length=2)
    ]
    assert r["validation_seconds"] >= 0 and r["serialization_seconds"] >= 0


def test_per_site_worker_checks_every_forward_and_keeps_row_identity(worker, tmp_path):
    worker.nh_publication_layout = "per_site"
    rows = [[1, 2], [3, 4]]
    worker.nh_begin("sites", ["a", "b"], rows, str(tmp_path / "wave"))
    forward(worker, ["a"], [rows[0]])
    forward(worker, ["b"], [rows[1]])
    worker.nh_seal("sites")
    receipt = poll(worker)[0]
    assert receipt["finite_validation"] == "device"
    assert load_file(receipt["files"]["L0_resid_post"]["path"])["L0_resid_post"].flatten().tolist() == [4, 7, 10, 13]
    assert sorted(worker.nh_pool.free) == [0, 1]


def test_active_site_reduction_waits_for_a_drained_boundary(worker, tmp_path):
    worker.nh_publication_layout = "per_site"
    worker.nh_set_sites(["L0_resid_post"])
    worker.nh_begin("reduced", ["a"], [[1, 2]], str(tmp_path / "wave"))
    with pytest.raises(RuntimeError, match="Drain"):
        worker.nh_set_sites(["L0_resid_pre"])
    forward(worker, ["a"], [[1, 2]])
    worker.nh_seal("reduced")
    receipt = poll(worker)[0]
    assert set(receipt["files"]) == {"token_ids", "L0_resid_post"}
    worker.nh_set_sites(["L0_resid_pre"])
    with pytest.raises(ValueError, match="subset"):
        worker.nh_set_sites(["wrong"])
