# Nemotron extraction 1.0

This package captures activations, a model's internal state as it processes tokens, from the official
[`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16)
checkpoint using Hugging Face Transformers with PyTorch or the vLLM inference engine.

A hook records activations at a chosen location, called a site.
In vLLM, hooks run inside graphics processor (GPU) worker processes.
The driver installs them through `collective_rpc` and submits token IDs.
Workers reconstruct residuals, preserve tensors before mutation, then copy them
to pinned host buffers and save Safetensors files. Transfers and writes can
overlap inference. Request IDs preserve sequence order, including repeats.

## Install and use

Python 3.10+; vLLM requires Linux and an NVIDIA GPU.

```bash
uv venv
uv pip install 'nemotron-extraction[hf,vllm,benchmark]==1.0.0'
```

```python
import torch
from nemotron_extraction import HookSite, VLLMExtractor

sites = [HookSite(12, "resid_post"), HookSite(12, "attn_out_prelinear")]
with VLLMExtractor(
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", sites,
    enable_chunked_prefill=True,
) as extractor:
    batch = extractor.extract(torch.tensor([[1, 2, 3]], dtype=torch.long))
    print(batch.activations["L12_resid_post"].shape)  # (3, hidden_width)
```

Both support `await extract_async(token_ids)` and
`stream_to_disk(token_batches, output_dir)`, yielding saved-file metadata.
At most two batches await saving. Layers are zero-based. Sites:
`resid_pre`, `resid_post`, `mamba_out`, `moe_out`, `attn_out`,
`attn_out_prelinear`; mixer-specific sites require a compatible layer.
Inputs are rectangular int64 batches without padding. CPU outputs have shape
`(batch × sequence_length, site_width)` in sequence-major order.

Detailed token checks and capture profiling are opt-in (`diagnostics=True`).
Request attribution, coverage and finite-output checks remain enabled.
`publication_threads=1` or `2` controls saving; the default is one.

Extraction requires eager execution, tensor parallelism one and disabled prefix
caching. Enable chunked prefill for this checkpoint; scheduler checks reject
partial prompts. Generated tokens are excluded.

## Validation

The benchmark freezes inputs and compares both backends sequentially:

```bash
uv run benchmarks/validate.py all
```

The activations are close, but only 3/18 sites passed all agreement checks per
language. Their interchangeability remains unvalidated.

The batch-8 sample contains 16,384 tokens/language from
FineWeb-Edu (English) and FineWeb2 (German/Spanish). Cosine measures direction
(1 means aligned). Normalised mean squared error (MSE) divides total squared
disagreement by total reference squared magnitude, retaining scale differences.
Zero means identical. Values are mean +/- population standard deviation (SD)
across 18 sites.

| Language | Mean cosine +/- SD | Normalised MSE +/- SD |
|---|---:|---:|
| English | 0.999002 +/- 0.001144 | 0.001879 +/- 0.002171 |
| German | 0.998623 +/- 0.001719 | 0.002650 +/- 0.003230 |
| Spanish | 0.998701 +/- 0.001619 | 0.002435 +/- 0.002989 |

The discrepancy remains unresolved. The default benchmark uses 1,000,000 tokens/language.
See the [report](report.pdf) and [source](report.tex).

Runs used Python 3.12, Hugging Face Transformers 4.57.6/PyTorch 2.8.0a0, and
vLLM 0.27.1/PyTorch 2.14.0a0.

For targeted investigations, `benchmarks/diagnose_divergence.py` traces operations;
`benchmarks/extraction_speed.py` measures direct saving at batches 8/64 with three
repetitions. Install the selected wheel with `uv` before comparing source changes.

## Contributing

Include a reproducer and preserve hook semantics, token ordering and attribution.
Run `uv sync --group dev`, `uv run pytest` and `uv build`.
Compile the report with `pdflatex report.tex`.

## Citation

Antonio Lobo-Santos, affiliated with DFKI. [BibTeX](CITATION.bib).
