from pathlib import Path
import sys

import pytest
import torch

BENCHMARKS = Path(__file__).parents[1] / "benchmarks"
sys.path.insert(0, str(BENCHMARKS))
import targeted_validation as targeted
from test_worker import worker, forward, poll


def test_exact_control_rejects_missing_shapes_and_nonfinite():
    values = {"site": torch.tensor([[1., 2.]])}
    assert targeted.exact(values, {"site": values["site"].clone()}) == {"site": True}
    assert targeted.exact(values, {"site": values["site"] + 1}) == {"site": False}
    with pytest.raises(ValueError, match="sites"):
        targeted.exact({}, {})
    with pytest.raises(ValueError, match="shape"):
        targeted.exact(values, {"site": torch.ones(2, 1)})
    with pytest.raises(ValueError, match="nonfinite"):
        targeted.exact(values, {"site": torch.full((1, 2), float("nan"))})


def test_overlap_union_does_not_double_count_and_handles_no_overlap():
    timeline = [{"name": "forward", "start": 1., "end": 4.},
                {"name": "forward", "start": 3., "end": 5.},
                {"name": "D2H", "start": 2., "end": 4.5},
                {"name": "D2H", "start": 4., "end": 6.},
                {"name": "write", "start": 6., "end": 7.}]
    assert targeted.overlap_seconds(timeline, "D2H") == 3.
    assert targeted.overlap_seconds(timeline, "write") == 0.
    assert targeted.overlap_seconds([], "write") == 0.


def test_control_cases_preserve_duplicates_reordering_and_short_tail():
    tokens = torch.arange(8192)
    short, full, tail = targeted.control_batches(tokens)
    assert short.shape == (4, 128) and full.shape == (4, 2048) and tail.shape == (1, 37)
    assert torch.equal(full[0], full[2])
    assert full[:, 0].tolist() == [4096, 0, 4096, 2048]
    assert torch.equal(short, full[:, :128]) and torch.equal(tail, full[:1, :37])


REFERENCE = '''
import a_training_dependency_that_does_not_exist
def _arg(args, kwargs, index, name):
    return kwargs[name] if name in kwargs else args[index]
def _residual(hidden, residual):
    return hidden.clone() if residual is None else hidden + residual
class ActivationHookExtension:
    def unused_transport(self):
        raise AssertionError("must be removed")
    def _sae_capture_resid_pre(self, slug, module, args, kwargs):
        self.sae_captured[slug] = _residual(_arg(args, kwargs, 1, "hidden_states"), _arg(args, kwargs, 2, "residual"))
    def _sae_capture_resid_post(self, slug, module, args, output):
        self.sae_captured[slug] = _residual(*output)
    def _sae_capture_output(self, slug, module, args, output):
        self.sae_captured[slug] = output.clone()
    def _sae_capture_input(self, slug, module, args, kwargs):
        self.sae_captured[slug] = _arg(args, kwargs, 0, "hidden_states").clone()
'''


def test_reference_loader_uses_original_callbacks_without_transport_imports():
    cls = targeted.reference_namespace(REFERENCE, worker=True)["ActivationHookExtension"]
    instance = cls()
    assert not hasattr(instance, "unused_transport")
    instance.sae_captured = {}
    hidden = torch.tensor([[2., 3.]], dtype=torch.bfloat16)
    running = torch.tensor([[4., 5.]], dtype=torch.bfloat16)
    instance._sae_capture_resid_pre("pre", None, (None, hidden, running), {})
    hidden.zero_()
    running.zero_()
    assert torch.equal(instance.sae_captured["pre"], torch.tensor([[6., 8.]]))


def test_worker_control_copies_match_after_reordering_and_can_be_disabled(worker, tmp_path):
    worker.__class__ = targeted.AuditWorker
    # Replace the boundary callback installed before changing the fixture's class.
    worker.nh_handles[-1].remove()
    worker.nh_handles[-1] = worker.model_runner.model.register_forward_hook(worker._nh_end)
    worker.ta_enable(REFERENCE)
    tokens = [[1, 2], [3, 4], [1, 2]]
    assert worker.nh_begin("audit", ["a", "b", "c"], tokens, str(tmp_path / "audit.safetensors"))
    forward(worker, ["b", "c", "a"], [tokens[1], tokens[2], tokens[0]])
    worker.nh_seal("audit")
    receipt = poll(worker)[0]
    assert all(receipt["same_forward_callbacks"][0].values())
    assert all(receipt["blocking_vs_async"].values())
    worker.ta_disable()
    assert worker.nh_begin("plain", ["d"], [[7]], str(tmp_path / "plain.safetensors"))
    forward(worker, ["d"], [[7]])
    worker.nh_seal("plain")
    assert "same_forward_callbacks" not in poll(worker)[0]
