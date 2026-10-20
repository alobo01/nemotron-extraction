import importlib
import sys
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from tiny_models import Model
from torch import nn

from nemotron_extraction import ActivationBatch, HFExtractor, HookSite, VLLMExtractor
from nemotron_extraction.hf import _cuda_device_index
from nemotron_extraction.hooks import install, representative_sites


def test_unindexed_cuda_device_resolves_for_set_device(monkeypatch):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)
    assert _cuda_device_index(torch.device("cuda")) == 3
    assert _cuda_device_index(torch.device("cuda:1")) == 1


def test_all_semantics_and_projection_widths():
    model = Model()
    sites = [
        HookSite(0, "resid_pre"),
        HookSite(0, "resid_post"),
        HookSite(0, "mamba_out"),
        HookSite(1, "moe_out"),
        HookSite(2, "attn_out"),
        HookSite(2, "attn_out_prelinear"),
    ]
    tokens = torch.tensor([[2, 3], [5, 7]])
    hidden = torch.stack([tokens.float(), tokens.float() + 1], -1).reshape(-1, 2)
    with HFExtractor(model, sites, device="cpu") as extractor:
        result = extractor.extract(tokens)
        assert torch.equal(result.token_ids, tokens)
        assert torch.equal(result.activations["L0_resid_pre"], hidden)
        assert torch.equal(result.activations["L0_mamba_out"], hidden * 2)
        assert torch.equal(result.activations["L0_resid_post"], hidden * 3)
        assert torch.equal(result.activations["L1_moe_out"], hidden * 3 + 3)
        assert torch.equal(result.activations["L2_attn_out"], hidden * 6 + 3)
        assert result.activations["L2_attn_out_prelinear"].shape == (4, 4)
        assert extractor.topology["L2_attn_out_prelinear"]["module_path"] == "model.layers.2.mixer.o_proj"
        # CPU outputs own their storage across subsequent extraction calls.
        extractor.extract(tokens + 10)
        assert torch.equal(result.activations["L0_resid_pre"], hidden)
        assert model.training
    assert not model.model.layers[0]._forward_pre_hooks
    extractor.close()
    with pytest.raises(RuntimeError, match="closed"):
        extractor.extract(tokens)


def test_forward_failure_cleans_captures_and_restores_mode():
    model = Model()
    with HFExtractor(model, [HookSite(0, "resid_post")], device="cpu") as extractor:
        model.model.fail = True
        with pytest.raises(RuntimeError, match="forward failure"):
            extractor.extract(torch.tensor([[1, 2]]))
        assert extractor.captured == {}
        assert model.training
        model.model.fail = False
        assert extractor.extract(torch.tensor([[1]])).activations["L0_resid_post"].shape == (1, 2)
    assert not model.model.layers[0]._forward_hooks


def test_invalid_topology_installation_rolls_back():
    model = Model()
    with pytest.raises(ValueError, match="compatible"):
        HFExtractor(model, [HookSite(0, "resid_post"), HookSite(0, "attn_out")], device="cpu")
    assert not model.model.layers[0]._forward_hooks
    assert len(representative_sites(model)) == 10


class SplitBlock(nn.Module):
    def forward(self, positions, hidden_states, residual):
        return hidden_states * 2, hidden_states + residual


def test_vllm_split_residual_and_input_mutation():
    model = nn.Module()
    model.model = nn.Module()
    model.model.layers = nn.ModuleList([SplitBlock()])
    captured = {}
    handles, _ = install(model, [HookSite(0, "resid_pre"), HookSite(0, "resid_post")], captured, worker=True)
    hidden, running = torch.tensor([[1.0, 2.0]]), torch.tensor([[4.0, 5.0]])
    model.model.layers[0](None, hidden, running)
    hidden.zero_()
    running.zero_()
    assert torch.equal(captured["L0_resid_pre"], torch.tensor([[5.0, 7.0]]))
    assert torch.equal(captured["L0_resid_post"], torch.tensor([[7.0, 11.0]]))
    for handle in handles:
        handle.remove()


@pytest.mark.parametrize("site", [(-1, "resid_post"), (True, "resid_post"), (0, "expert")])
def test_invalid_sites(site):
    with pytest.raises(ValueError):
        HookSite(*site)


def test_rejects_padding_like_shapes_and_nonfinite():
    with HFExtractor(Model(), [HookSite(0, "resid_post")], device="cpu") as extractor:
        for tokens in (torch.ones(2, 3), torch.ones(2, dtype=torch.long), torch.empty(0, 2, dtype=torch.long)):
            with pytest.raises(ValueError, match="rectangular"):
                extractor.extract(tokens)
    with pytest.raises(ValueError, match="nonfinite"):
        ActivationBatch(torch.tensor([[1]]), {"x": torch.tensor([[float("nan")]])})


def test_vllm_driver_transport_cleanup_and_installed_worker(monkeypatch, tmp_path):
    instances = []

    class Engine:
        def __init__(self, args):
            self.kwargs = kwargs = vars(args)
            module, name = kwargs["worker_extension_cls"].rsplit(".", 1)
            assert getattr(importlib.import_module(module), name)
            self.llm_engine = SimpleNamespace(model_executor=SimpleNamespace(shutdown=self.shutdown))
            self.stopped = False
            instances.append(self)

        @classmethod
        def from_engine_args(cls, args):
            return cls(args)

        async def generate(self, prompt, params, request_id):
            yield None

        async def abort(self, request_id):
            pass

        async def collective_rpc(self, method, args=()):
            if method == "nh_install":
                self.sink = args[1]
                return [{"L0_resid_post": {"module_path": "model.layers.0"}}]
            if method == "nh_begin":
                self.wave, _, tokens, self.path = args
                self.tokens = torch.tensor(tokens)
                return [True]
            if method == "nh_seal":
                save_file({"token_ids": self.tokens, "L0_resid_post": self.tokens.reshape(-1, 1).float()}, self.path)
                self.ready = getattr(self, "ready", []) + [
                    {
                        "path": self.path,
                        "wave_id": self.wave,
                        "topology": {},
                        "peak_gpu_memory_bytes": 123,
                        "peak_gpu_reserved_bytes": 123,
                        "peak_device_memory_bytes": 123,
                        "total_device_memory_bytes": 1000,
                        "peak_device_memory_fraction": 0.123,
                        "actual_forward_batches": [len(self.tokens)],
                        "worker_copy_seconds": 0.0,
                        "write_seconds": 0.0,
                        "worker_forward_seconds": 0.0,
                        "validation_seconds": 0.0,
                        "serialization_seconds": 0.0,
                    }
                ]
                return [True]
            if method == "nh_poll":
                ready = getattr(self, "ready", [])
                self.ready = []
                return [ready]
            return [None]

        def shutdown(self):
            self.stopped = True

    monkeypatch.setitem(
        sys.modules,
        "vllm",
        SimpleNamespace(AsyncEngineArgs=lambda **k: SimpleNamespace(**k), SamplingParams=lambda **k: k),
    )
    monkeypatch.setitem(sys.modules, "vllm.v1.engine.async_llm", SimpleNamespace(AsyncLLM=Engine))
    with VLLMExtractor("fixture", [HookSite(0, "resid_post")], sink_dir=tmp_path, sequences_per_call=2) as extractor:
        tokens = torch.tensor([[1, 2], [3, 4], [1, 2]])
        result = extractor.extract(tokens)
        assert result.activations["L0_resid_post"].flatten().tolist() == tokens.flatten().tolist()
        assert extractor.peak_gpu_memory_bytes == 123
        assert extractor.peak_device_memory_fraction == 0.123
    assert instances[0].stopped
    assert not list(tmp_path.iterdir())
    assert instances[0].kwargs["enable_prefix_caching"] is False
    assert instances[0].kwargs["enforce_eager"] is True
    assert instances[0].kwargs["kv_cache_memory_bytes"] is None
    with VLLMExtractor(
        "fixture", [HookSite(0, "resid_post")], sink_dir=tmp_path, sequences_per_call=2, enable_chunked_prefill=True
    ):
        assert instances[1].kwargs["enable_chunked_prefill"] is True


def test_device_extraction_returns_trainable_storage(monkeypatch):
    import nemotron_extraction.hf as hf

    # Only the CUDA-only output validator is substituted; the forward path is real.
    monkeypatch.setattr(
        hf, "DeviceActivationBatch", lambda tokens, rows: SimpleNamespace(token_ids=tokens, activations=rows)
    )
    with HFExtractor(Model(), [HookSite(0, "resid_post")], device="cpu") as extractor:
        batch = extractor.extract_device(torch.tensor([[1, 2, 3]]))
    rows = batch.activations["L0_resid_post"]
    assert not torch.is_inference(rows)
    weights = torch.ones(rows.shape[1], 3, requires_grad=True)
    (rows @ weights).square().mean().backward()
    assert weights.grad is not None and torch.isfinite(weights.grad).all()


@pytest.mark.parametrize("factory", [HFExtractor, VLLMExtractor])
@pytest.mark.parametrize("threads", [0, 3, True])
def test_writer_limit_rejected_before_loading(factory, threads):
    with pytest.raises(ValueError, match="publication_threads"):
        factory("unused", [HookSite(0, "resid_post")], publication_threads=threads)
