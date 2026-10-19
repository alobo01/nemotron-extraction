"""Check source isolation, row assembly, fair batching and metric validation."""
from pathlib import Path
import sys

import pytest
import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).parents[1] / "benchmarks"))
import compare_repositories as comparison


def snapshot(directory):
    (directory / "attention.py").write_text('def output_projection(m): return m\n')
    (directory / "worker.py").write_text('''
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
''')


def test_snapshot_preserves_definitions_and_constants_without_training_imports(tmp_path):
    snapshot(tmp_path)
    scope = comparison.sibling_namespace(tmp_path)
    assert scope["MIXER_KINDS"] == {"moe_out":"MoE"}
    assert hasattr(scope["ActivationHookExtension"], "sae_save")
    assert scope["output_projection"](42) == 42


def test_timing_inputs_have_identical_counts_and_full_requested_shapes():
    tokens = torch.arange(131072)
    for batch_size in (8,16,32,64,128):
        batches = comparison.cases(tokens,batch_size,512,131072)
        assert all(b.shape == (batch_size,512) for b in batches)
        assert torch.equal(torch.cat(batches).flatten(),tokens)
        assert len(batches) >= 2
    with pytest.raises(ValueError,match="full requested"):
        comparison.cases(tokens,128,512,65537)


def test_shard_loading_checks_order_and_handles_original_fill_layout(tmp_path):
    site = comparison.HookSite(0,"resid_post")
    tokens = torch.tensor([[1,2],[1,2]])
    values = torch.arange(8,dtype=torch.bfloat16).reshape(4,2)
    save_file({"token_ids":tokens},str(tmp_path / "token_ids.safetensors"))
    save_file({"activations":values},str(tmp_path / f"{site.key}.safetensors"))
    receipt = {"mode":"sibling_fills","path":str(tmp_path)}
    assert torch.equal(comparison.load_capture(receipt,tokens,[site],tmp_path)[site.key],values)
    with pytest.raises(ValueError,match="token order"):
        comparison.load_capture(receipt,tokens+1,[site],tmp_path)


def test_comparison_rejects_short_shards_and_records_exact_agreement(tmp_path):
    site = comparison.HookSite(0,"resid_post")
    tokens = torch.tensor([[1,2]])
    path = tmp_path / "capture.safetensors"
    save_file({"token_ids":tokens,site.key:torch.ones(2,3,dtype=torch.bfloat16)},str(path))
    receipts = [{"path":str(path)}]
    result = comparison.compare_receipts(receipts,receipts,[tokens],[site],tmp_path,tmp_path/"sample.safetensors")
    assert result[site.key]["passed"]
    assert result[site.key]["normalized_mse"] == 0
    assert result[site.key]["cosine_std"] == 0
    save_file({"token_ids":tokens,site.key:torch.ones(1,3,dtype=torch.bfloat16)},str(path))
    with pytest.raises(ValueError,match="row count"):
        comparison.compare_receipts(receipts,receipts,[tokens],[site],tmp_path,tmp_path/"sample.safetensors")
