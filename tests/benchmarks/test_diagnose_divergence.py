import asyncio
import json
from types import SimpleNamespace

import diagnose_divergence as diagnostic
import torch
from benchlib.divergence import analyse, boundary_order, pair_metrics
from safetensors.torch import load_file, save_file
from torch import nn

from nemotron_extraction import HookSite


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


def test_case_definitions_keep_anchor_identity_and_short_tail():
    rows = torch.arange(64 * 2048).reshape(64, 2048)
    for name, tokens, anchor in diagnostic.cases(rows):
        expected = rows[0, :37] if name == "short37" else (rows[0, :128] if name == "short128" else rows[0])
        assert torch.equal(tokens[anchor], expected)
    assert boundary_order("L02_mixer_input") < boundary_order("L02_mixer_out")


def test_comparison_selects_first_failure_and_saves_worst_samples(tmp_path):
    for engine in ("transformers", "vllm"):
        d = tmp_path / engine
        d.mkdir()
        h = torch.ones(37, 4)
        v = h if engine == "transformers" else h * 2
        save_file(
            {
                "token_ids": torch.arange(37),
                "embedding": h.clone(),
                "L00_resid_pre": h.clone(),
                "L00_mixer_input": h.clone(),
                "L00_mixer_out": v.clone(),
                "L00_resid_post": v.clone(),
            },
            str(d / "alone.safetensors"),
        )
    analyse(SimpleNamespace(output=tmp_path))
    result = json.loads((tmp_path / "analysis.json").read_text())
    assert result["first_failing_boundary"] == "L00_mixer_out"
    assert result["replay"]["key"] == "L00_mixer_input"
    assert result["gates_unchanged"]


def test_original_transformers_persistence_matches_same_forward_oracle_and_reuses_buffers(tmp_path):
    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Block()])

        def forward(self, input_ids, use_cache=False):
            return self.layers[0](input_ids.float().unsqueeze(-1).repeat(1, 1, 4))

    class Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Decoder()

    model = Wrapper()
    with diagnostic.IntegrityHFExtractor(
        model, [HookSite(0, "resid_pre"), HookSite(0, "resid_post")], device="cpu"
    ) as extractor:
        rows = torch.tensor([[1, 2, 3], [4, 5, 6], [1, 2, 3]])

        async def collect():
            return [r async for r in extractor.stream_to_disk([rows, rows.flip(0), rows[:1, :2]], tmp_path)]

        receipts = asyncio.run(collect())
        assert all(r["diagnostic_transport_exact"] for r in receipts)
        assert {r["diagnostic_slot"] for r in receipts} == {0, 1}
        assert torch.equal(load_file(receipts[-1]["path"])["token_ids"], rows[:1, :2])
        assert sorted(extractor.pool.free) == [0, 1]
    assert not model.model._forward_hooks


def test_anchor_rule_is_reported_alongside_unchanged_strict_gates():
    reference = torch.ones(100, 8)
    candidate = reference.clone()
    candidate[-1] *= -1
    result = pair_metrics(reference, candidate)
    assert result["rows_cosine_at_least_0_990"] == 99
    assert result["passes_anchor_99_percent_rule"]
    assert not result["strict_mean_cosine_pass"]
    assert not result["strict_minimum_cosine_pass"]
