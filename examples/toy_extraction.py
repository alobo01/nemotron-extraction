"""Extract activations from a three-layer stand-in for Nemotron on a CPU, in memory and streamed to disk.

The stand-in has the same layout the hooks look for (`model.layers[i].mixer`, with Mamba, MoE and attention
mixers), so every site kind works without downloading the real checkpoint. Swap `ToyNemotron()` for the model ID
and `HFExtractor` for `VLLMExtractor` to run the same code on Nemotron 3 Nano.

    python examples/toy_extraction.py [OUT_DIR]
"""

import asyncio
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import nn

from nemotron_extraction import HFExtractor, HookSite

WIDTH = 8


class ToyMamba(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_proj = nn.Linear(WIDTH, WIDTH)

    def forward(self, hidden_states):
        return torch.tanh(self.in_proj(hidden_states))


class ToyMoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = nn.ModuleList(nn.Linear(WIDTH, WIDTH) for _ in range(2))

    def forward(self, hidden_states):
        return sum(expert(hidden_states) for expert in self.experts) / len(self.experts)


class ToyAttention(nn.Module):
    """Causal self-attention whose output projection makes the `attn_out_prelinear` site available."""

    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(WIDTH, 3 * WIDTH)
        self.o_proj = nn.Linear(WIDTH, WIDTH)

    def forward(self, hidden_states):
        q, k, v = self.qkv(hidden_states).chunk(3, dim=-1)
        mixed = nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o_proj(mixed)


class Block(nn.Module):
    def __init__(self, mixer):
        super().__init__()
        self.mixer = mixer

    def forward(self, hidden_states):
        return hidden_states + self.mixer(hidden_states)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(100, WIDTH)
        self.layers = nn.ModuleList([Block(ToyMamba()), Block(ToyMoE()), Block(ToyAttention())])

    def forward(self, input_ids, use_cache=False):
        hidden = self.embed(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden


class ToyNemotron(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Decoder()


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/toy_extraction")
    torch.manual_seed(0)
    sites = [HookSite(0, "mamba_out"), HookSite(1, "moe_out"), HookSite(2, "attn_out_prelinear"),
             HookSite(2, "resid_post")]
    tokens = torch.randint(0, 100, (2, 5))  # [batch, sequence], int64, no padding

    with HFExtractor(ToyNemotron(), sites, device="cpu", publication_layout="per_site") as extractor:
        batch = extractor.extract(tokens)
        for key, rows in batch.activations.items():
            print(f"{key:24s} {tuple(rows.shape)}")  # one row per input token: [batch * sequence, width]

        async def stream():
            # Two waves of the same shape; each becomes a directory of per-site files with a checksummed receipt.
            return [receipt async for receipt in extractor.stream_to_disk([tokens, tokens + 1], out)]

        for receipt in asyncio.run(stream()):
            saved = load_file(receipt["files"]["L2_resid_post"]["path"])["L2_resid_post"]
            files = len(receipt["files"])
            print(f"wave {receipt['index']}: {receipt['path']} ({files} files, rows {tuple(saved.shape)})")
    assert torch.equal(load_file(f"{out}/000000/L2_resid_post.safetensors")["L2_resid_post"],
                       batch.activations["L2_resid_post"])


if __name__ == "__main__":
    main()
