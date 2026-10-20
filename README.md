# Nemotron extraction

Capture activations (a model's internal state for each token) from the official
[`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16)
checkpoint with Hugging Face Transformers (PyTorch) or the vLLM inference engine.

## Install

Python 3.10+. vLLM requires Linux and an NVIDIA GPU.

```bash
uv venv
uv pip install 'nemotron-extraction[hf,vllm]==1.1.0'
```

## Use

A hook records activations at a *site*: a zero-based layer and a kind.

```python
import torch
from nemotron_extraction import HFExtractor, HookSite, VLLMExtractor

model = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
sites = [HookSite(12, "resid_post"), HookSite(12, "attn_out_prelinear")]
tokens = torch.tensor([[1, 2, 3]], dtype=torch.long)

with VLLMExtractor(model, sites, enable_chunked_prefill=True) as extractor:
    batch = extractor.extract(tokens)
    print(batch.activations["L12_resid_post"].shape)  # (3, hidden_width)

with HFExtractor(model, sites) as extractor:  # PyTorch reference, same interface
    batch = extractor.extract(tokens)
```

Both extractors also provide:

- `await extract_async(token_ids)`.
- `stream_to_disk(token_batches, output_dir)`, an async iterator of saved-shard receipts. At most two batches
  await saving; `publication_threads=1` (default) or `2` sets how many write in parallel.

`HFExtractor.extract_device(token_ids)` instead returns the activations on the GPU for live training.

| Site kind | Captures |
|---|---|
| `resid_pre`, `resid_post` | Residual stream entering / leaving the layer |
| `mamba_out`, `moe_out`, `attn_out` | Mixer output; the layer must have that mixer |
| `attn_out_prelinear` | Attention output before its output projection |

Inputs are rectangular int64 `[batch, length]` token IDs without padding. Outputs are CPU tensors of shape
`(batch × length, width)` in sequence-major order: all positions of the first sequence, then the second.
Generated tokens are excluded.

vLLM extraction runs eagerly with tensor parallelism one and prefix caching disabled. Enable chunked prefill
for this checkpoint; scheduler checks reject partial prompts. Request attribution, coverage and finite-value
checks always run. Per-token checks and capture profiling are opt-in with `diagnostics=True`.

### How the vLLM backend works

Hooks run inside the GPU worker processes. The driver installs them through `collective_rpc` and submits
token IDs. Workers reconstruct residuals, copy tensors before vLLM mutates them, and transfer them to pinned
host buffers. The workers then write Safetensors shards, and those writes overlap inference. Request IDs
preserve sequence order, including repeated prompts.

## Validation

`benchmarks/validate.py` freezes inputs and compares both backends, one after the other:

```bash
uv run benchmarks/validate.py all
```

The activations are close, but only 3 of 18 sites passed all agreement checks per language. The two backends
are therefore not validated as interchangeable.

The batch-8 sample has 16,384 tokens per language from FineWeb-Edu (English) and FineWeb2 (German, Spanish).
Cosine measures direction (1 means aligned). Normalised mean squared error (MSE) divides total squared
disagreement by total reference squared magnitude, so scale differences count. Zero means identical. Values
are mean ± population standard deviation across the 18 sites.

| Language | Mean cosine ± SD | Normalised MSE ± SD |
|---|---:|---:|
| English | 0.999002 ± 0.001144 | 0.001879 ± 0.002171 |
| German | 0.998623 ± 0.001719 | 0.002650 ± 0.003230 |
| Spanish | 0.998701 ± 0.001619 | 0.002435 ± 0.002989 |

The cause of the discrepancy is unresolved. The default benchmark uses 1,000,000 tokens per language. Runs
used Python 3.12, Transformers 4.57.6 with PyTorch 2.8.0a0, and vLLM 0.27.1 with PyTorch 2.14.0a0. See the
[report](docs/report.pdf) ([source](docs/report.tex)) and [benchmarks/README.md](benchmarks/README.md) for
the other experiments.

## Repository layout

```
src/nemotron_extraction/   the installed package
  hf.py, vllm.py             the two extractors
  extractor.py               state and metrics they share
  hooks.py                   Nemotron topology and hook installation
  worker.py                  vLLM worker extension
  transport.py               pinned buffers, timing and atomic shard writes
  types.py                   HookSite, ActivationBatch and input validation
benchmarks/                five entry-point scripts; benchlib/ holds their shared code (not installed)
tests/                     package tests; tests/benchmarks/ and tests/smoke/ cover the rest
docs/                      validation report
```

## Contributing

Include a reproducer and preserve hook semantics, token ordering and attribution.

```bash
uv sync --group dev
uv run pytest
uvx ruff check src benchmarks tests && uvx ruff format --check src benchmarks tests
uv build
```

Compile the report with `pdflatex docs/report.tex`.

## Citation and license

Antonio Lobo-Santos, affiliated with DFKI ([BibTeX](CITATION.bib)). MIT licensed; see [LICENSE](LICENSE).
