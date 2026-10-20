"""benchlib diagnostics must distinguish computation, identity and transport."""

import hashlib
from types import SimpleNamespace

import pytest
import torch
from benchlib.instrumentation import overlap_seconds
from benchlib.moe import manual_nemotron_moe, manual_sigmoid_router, routing_report
from benchlib.precision import checkpoint_source_reference, ieee_precision, precision_settings
from benchlib.sibling import configured_chunked_prefill, configured_kv_cache_bytes, sibling_namespace
from benchlib.trace import TraceHooks, runtime_config
from benchlib.workers import DiagnosticWorker
from precision_validation import gate_passed
from torch import nn


class Mixer(nn.Module):
    def forward(self, hidden_states):
        return hidden_states * 2


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.Identity()
        self.mixer = Mixer()

    def forward(self, hidden_states):
        return hidden_states + self.mixer(self.norm(hidden_states))


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(64, 4)
        self.model.layers = nn.ModuleList([Block(), Block()])

    def forward(self, input_ids):
        x = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            x = layer(x)
        return x


def test_anchor_trace_survives_mutation_replays_common_input_and_cleans_up():
    model = Model()
    hooks = TraceHooks(model)
    tokens = torch.tensor([[1, 2, 3], [4, 5, 6]])
    hooks.span = (3, 6)
    model(tokens)
    embedding = hooks.values["embedding"].clone()
    assert embedding.shape == (3, 4)
    assert torch.equal(hooks.values["L00_resid_post"], embedding * 3)
    with torch.no_grad():
        model.model.embed_tokens.weight.zero_()
    assert torch.equal(hooks.values["embedding"], embedding)
    hooks.reset()
    hooks.span = (0, 3)
    hooks.replay = dict(layer=0, operation="mixer", input=torch.ones(3, 4))
    model(tokens[:1])
    assert torch.equal(hooks.values["L00_mixer_out"], torch.full((3, 4), 2.0))
    hooks.close()
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())


def test_runtime_config_converts_torch_version_to_rpc_safe_builtin_string():
    assert type(runtime_config(Model())["torch_version"]) is str


def test_scheduler_records_reordered_duplicates_and_rejects_partial_prefill():
    worker = DiagnosticWorker()
    worker.diag_trace = SimpleNamespace(span=None)
    worker.diag_case = dict(requests={"a": [1, 2], "b": [3, 4], "c": [1, 2]}, anchor="a", scheduler=[], covered=set())
    batch = SimpleNamespace(
        req_ids=["b-01234567", "c-01234567", "a-01234567"],
        token_ids_cpu=torch.tensor([[3, 4], [1, 2], [1, 2]]),
        num_computed_tokens_cpu=[0, 0, 0],
    )
    worker.model_runner = SimpleNamespace(
        input_batch=batch, query_start_loc=SimpleNamespace(cpu=torch.tensor([0, 2, 4, 6]))
    )
    worker._diag_inputs(None, (), {})
    assert worker.diag_trace.span == (4, 6)
    assert worker.diag_case["covered"] == {"a", "b", "c"}
    assert len(worker.diag_case["scheduler"]) == 3
    worker.diag_case["covered"].clear()
    batch.num_computed_tokens_cpu = [1, 0, 0]
    with pytest.raises(RuntimeError, match="Partial prefill"):
        worker._diag_inputs(None, (), {})


def test_same_forward_comparison_detects_disagreement_before_transport():
    worker = DiagnosticWorker()
    worker.diag_case = {"integrity": True, "equality": []}
    worker.diag_package = {"site": torch.ones(2, 3)}
    worker.diag_sibling = SimpleNamespace(sae_captured={"site": torch.zeros(2, 3)})
    with pytest.raises(RuntimeError, match="callbacks differ"):
        worker._diag_equal(None, None, None)


def test_router_report_uses_observed_ids_and_reports_selection_margin():
    logits = torch.tensor([[3.0, 2.0, 1.0, 0.0], [0.0, 1.0, 2.0, 3.0]])
    bias = torch.tensor([0.0, 0.0, 0.0, 0.0])
    ids = torch.tensor([[0, 1], [3, 2]])
    report = routing_report(logits, bias, ids)
    assert report["unconstrained_set_agreement"].tolist() == [True, True]
    assert torch.all(report["global_sixth_seventh_margin"] > 0)


def test_manual_sigmoid_router_uses_correction_for_selection_only():
    x = torch.tensor([[1.0, 0.0]])
    weight = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, 0.0]])
    correction = torch.tensor([0.0, 2.0, 0.0])
    result = manual_sigmoid_router(x, weight, correction, top_k=2, routed_scale=2.5)
    assert set(result["ids"][0].tolist()) == {0, 1}
    expected = result["scores"].gather(1, result["ids"])
    expected = expected / expected.sum(1, keepdim=True) * 2.5
    assert torch.allclose(result["weights"], expected)
    assert not torch.equal(result["weights"], torch.full_like(result["weights"], 1.25))


def test_manual_nemotron_moe_matches_explicit_non_gated_equations():
    torch.manual_seed(19)
    x = torch.randn(5, 4, dtype=torch.float32)
    up = torch.randn(3, 6, 4)
    down = torch.randn(3, 4, 6)
    shared_up = torch.randn(7, 4)
    shared_down = torch.randn(4, 7)
    ids = torch.tensor([[0, 1], [1, 2], [2, 0], [0, 2], [1, 0]])
    weights = torch.rand(5, 2)
    weights = weights / weights.sum(1, keepdim=True)
    result = manual_nemotron_moe(
        x, ids, weights, up, down, shared_up_weight=shared_up, shared_down_weight=shared_down, routed_scale=2.5
    )
    ref = torch.zeros_like(x)
    for row in range(len(x)):
        for slot in range(ids.shape[1]):
            expert = ids[row, slot]
            mid = torch.nn.functional.linear(x[row : row + 1], up[expert])
            out = torch.nn.functional.linear(torch.relu(mid).square(), down[expert])
            ref[row] += out[0] * (weights[row, slot] * 2.5)
    ref = ref + torch.nn.functional.linear(torch.relu(torch.nn.functional.linear(x, shared_up)).square(), shared_down)
    assert torch.allclose(result["output"], ref)


def test_reference_accumulation_preserves_expert_order_under_cancellation():
    x = torch.ones(1, 1)
    up = torch.ones(3, 1, 1)
    down = torch.tensor([1e8, -1e8, 1.0]).reshape(3, 1, 1)
    # Slot order gives (1e8 + 1) - 1e8 == 0; expert order gives 1.
    result = manual_nemotron_moe(x, torch.tensor([[0, 2, 1]]), torch.ones(1, 3), up, down)
    assert torch.equal(result["output"], torch.ones(1, 1))


def test_ieee_precision_restores_state_after_failure():
    previous = precision_settings()
    with pytest.raises(RuntimeError, match="controlled failure"):
        with ieee_precision() as current:
            assert current["bf16_reduced_reduction"] is False
            if current["api"] == "modern":
                assert all(current[k] == "ieee" for k in ("global", "matmul", "cudnn", "conv", "rnn"))
            else:
                assert not current["matmul_tf32"] and not current["cudnn_tf32"]
            raise RuntimeError("controlled failure")
    assert precision_settings() == previous


def test_reference_source_checksum_and_missing_classes_fail_closed(tmp_path):
    source = tmp_path / "reference.py"
    source.write_text('raise RuntimeError("imports must not execute")\n')
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="checksum"):
        checkpoint_source_reference(source, "0" * 64, {})
    with pytest.raises(ValueError, match="class"):
        checkpoint_source_reference(source, digest, {})


def test_source_loader_uses_frozen_classes_without_executing_imports(tmp_path):
    source = tmp_path / "reference.py"
    source.write_text("""
raise RuntimeError("Mamba imports must not execute")
class NemotronHMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
class NemotronHTopkRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
class NemotronHMOE(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.weight = nn.Parameter(torch.empty(config.hidden_size))
""")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    model = checkpoint_source_reference(source, digest, dict(hidden_size=7))
    assert model.weight.device.type == "meta"
    assert model.weight.shape == (7,)
    assert model.layer_idx == 1


def test_final_validation_requires_every_precision_and_probe_check():
    result = dict(strict_pass=True, exact_router_logits=True, expert_sets_equal=True, persisted_samples_exact=True)
    assert gate_passed(result)
    for key in result:
        assert not gate_passed(dict(result, **{key: False}))


def snapshot(directory):
    (directory / "attention.py").write_text("def output_projection(m): return m\n")
    (directory / "worker.py").write_text("""
import missing_training_package
MIXER_KINDS = {"moe_out": "MoE"}
class ActivationHookExtension:
    def sae_install_hooks(self): pass
    def _sae_end_step(self): pass
    def sae_save(self): pass
    def sae_flush_saves(self): pass
    def sae_flush(self): pass
def assemble_rows(steps, tokens, keys):
    return {k: torch.cat([step[k] for step in steps]) for k in keys}
""")


def test_snapshot_preserves_definitions_and_constants_without_training_imports(tmp_path):
    snapshot(tmp_path)
    scope = sibling_namespace(tmp_path)
    assert scope["MIXER_KINDS"] == {"moe_out": "MoE"}
    assert hasattr(scope["ActivationHookExtension"], "sae_save")
    assert scope["output_projection"](42) == 42


def test_cache_size_is_automatic_unless_explicit(monkeypatch):
    monkeypatch.delenv("NH_KV_CACHE_BYTES", raising=False)
    assert configured_kv_cache_bytes() is None
    monkeypatch.setenv("NH_KV_CACHE_BYTES", "4096")
    assert configured_kv_cache_bytes() == 4096
    monkeypatch.delenv("NH_ENABLE_CHUNKED_PREFILL", raising=False)
    assert configured_chunked_prefill() is True
    monkeypatch.setenv("NH_ENABLE_CHUNKED_PREFILL", "false")
    assert configured_chunked_prefill() is False


def test_overlap_union_does_not_double_count_and_handles_no_overlap():
    timeline = [
        {"name": "forward", "start": 1.0, "end": 4.0},
        {"name": "forward", "start": 3.0, "end": 5.0},
        {"name": "D2H", "start": 2.0, "end": 4.5},
        {"name": "D2H", "start": 4.0, "end": 6.0},
        {"name": "write", "start": 6.0, "end": 7.0},
    ]
    assert overlap_seconds(timeline, "D2H") == 3.0
    assert overlap_seconds(timeline, "write") == 0.0
    assert overlap_seconds([], "write") == 0.0
