from types import SimpleNamespace

import pytest
import scaling
import torch
from test_worker import poll, schedule, worker  # noqa: F401  (fixture)


def test_forward_only_waves_time_the_forward_and_write_nothing(worker, tmp_path):  # noqa: F811
    worker.__class__ = scaling.ScalingWorker
    worker.nh_forward_only(True)
    rows = [[1, 2, 3], [4, 5, 6]]
    worker.nh_begin("fwd", ["a", "b"], rows, str(tmp_path / "fwd"))
    schedule(worker, [("a", rows[0], 0, 3), ("b", rows[1], 0, 1)])
    schedule(worker, [("b", rows[1], 1, 2)])
    worker.nh_seal("fwd")
    receipt = poll(worker)[0]
    assert receipt["path"] is None and receipt["tokens"] == 6
    assert receipt["worker_forward_seconds"] == pytest.approx(0.002)
    assert receipt["actual_forward_batches"] == [2, 1]
    assert not list(tmp_path.iterdir()) and sorted(worker.nh_pool.free) == [0, 1]
    worker.nh_begin("open", ["c"], rows[:1], str(tmp_path / "open"))
    with pytest.raises(RuntimeError, match="Drain"):
        worker.nh_forward_only(False)


def test_two_waves_come_from_the_start_of_the_stream_repeated_when_short():
    first, second = scaling.wave_inputs(torch.arange(100), 5, 2)
    assert first.tolist() == [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]] and second[0, 0] == 10
    first, second = scaling.wave_inputs(torch.arange(7), 5, 2)
    assert second.flatten().tolist() == [3, 4, 5, 6, 0, 1, 2, 3, 4, 5]


def test_only_pytorch_cells_above_its_forward_limit_are_skipped():
    hf = SimpleNamespace(config="hf", wave_token_cap=1 << 20)
    assert scaling.skip_reason(hf, 208_000) is None
    assert "PyTorch forward" in scaling.skip_reason(hf, 262_144)
    assert scaling.skip_reason(SimpleNamespace(config="vllm-whole", wave_token_cap=1 << 20), 262_144) is None
    assert "token cap" in scaling.skip_reason(SimpleNamespace(config="hf", wave_token_cap=1000), 2000)
