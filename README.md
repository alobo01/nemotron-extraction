# Nemotron extraction 1.0

Activations describe a model's internal state as it processes tokens.
This package captures them from the official
[`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16)
checkpoint through two backends: Hugging Face Transformers, running with PyTorch,
and vLLM, an inference engine.

A hook captures activations at a chosen location (site).
In vLLM, hooks run inside worker processes on the graphics processor (GPU).
The driver installs hooks through `collective_rpc` and submits token IDs.
Workers copy captures into pinned host buffers on a separate CUDA stream and
save Safetensors files, allowing transfers and writes to overlap inference.
Request IDs preserve sequence order, including repeats.

Hooks reconstruct residuals and clone tensors before later operations overwrite them.

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
) as extractor:
    batch = extractor.extract(torch.tensor([[1, 2, 3]], dtype=torch.long))
    print(batch.activations["L12_resid_post"].shape)  # (3, hidden_width)
```

Both support `await extract_async(token_ids)` and
`stream_to_disk(token_batches, output_dir)`, yielding saved-file metadata.
At most two batches await saving. Layers are zero-based. Sites:
`resid_pre`, `resid_post`, `mamba_out`, `moe_out`, `attn_out`,
`attn_out_prelinear`; mixer-specific sites require a compatible layer.
Inputs are rectangular int64 batches without padding; outputs in host memory have shape
`(batch × sequence_length, site_width)` in sequence-major order.
Use separate calls for different lengths.

Extraction requires eager execution, tensor parallelism of one, and disabled
prefix caching and chunked prefill. Only prompt activations are captured.

## Validation

One script freezes inputs, compares both backends sequentially for English,
German and Spanish, and saves JSON results and Markdown reports.
uv installs the package and dependencies automatically:

```bash
uv run benchmarks/validate.py all
```

Hugging Face Transformers and vLLM produce close activations, but only 3/18 sites
passed all agreement checks per language. Interchangeability for sparse autoencoder training
remains unvalidated.

The table summarises a batch-8 validation sample with 16,384 tokens/language from
FineWeb-Edu (English) and FineWeb2 (German/Spanish). Cosine measures direction
(1 means aligned). Normalised mean squared error (MSE) divides squared error by
the reference's squared norm. Both backends share the Hugging Face Transformers
reference root mean square as their scale.
It retains magnitude differences; 0 means identical. Values are mean +/- population standard deviation (SD)
across 18 equally weighted sites.

| Language | Mean cosine +/- SD | Normalised MSE +/- SD |
|---|---:|---:|
| English | 0.999002 +/- 0.001144 | 0.001879 +/- 0.002171 |
| German | 0.998623 +/- 0.001719 | 0.002650 +/- 0.003230 |
| Spanish | 0.998701 +/- 0.001619 | 0.002435 +/- 0.002989 |

The discrepancy remains unresolved. The default benchmark uses exactly
1,000,000 tokens per language.
See the [report](report.pdf) and [source](report.tex).

Runs used Python 3.12, Hugging Face Transformers 4.57.6/PyTorch 2.8.0a0, and
vLLM 0.27.1/PyTorch 2.14.0a0.

## Contributing

Include a reproducer and preserve hook semantics, token ordering and attribution.
Run `uv sync --group dev`, `uv run pytest` and `uv build`.
Compile the report with `pdflatex report.tex`.

## Citation

Antonio Lobo-Santos, DFKI. Download [BibTeX](CITATION.bib):

```bibtex
@misc{lobo_santos2026nemotron_extraction,
  author = {Lobo-Santos, Antonio},
  title = {{Nemotron Extraction}},
  year = {2026},
  howpublished = {Software, version 1.0.0},
  note = {DFKI}
}
```
