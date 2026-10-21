# Nemotron extraction

Capture token-aligned activations from NVIDIA's [Nemotron 3 Nano 30B A3B BF16 checkpoint](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16) with either Hugging Face Transformers (PyTorch) or vLLM. Both extractors share the same interface and save one row per input token at each selected model site.

## Install

Python 3.10 or later is required. vLLM needs Linux and a supported NVIDIA GPU. Install the backend extras you plan to use:

```bash
uv venv
uv pip install 'nemotron-extraction[hf,vllm]==1.1.0'
```

Use `[hf]` or `[vllm]` alone to install just one backend.

## Quick start

```python
import torch
from nemotron_extraction import HookSite, VLLMExtractor

model = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
sites = [HookSite(12, "resid_post"), HookSite(12, "attn_out_prelinear")]
token_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

with VLLMExtractor(model, sites, enable_chunked_prefill=True) as extractor:
    batch = extractor.extract(token_ids)

print(batch.activations["L12_resid_post"].shape)  # (3, hidden_width)
```

For the PyTorch reference, replace `VLLMExtractor` with `HFExtractor` and install the `[hf]` extra. The model can also be supplied as a caller-owned PyTorch module to `HFExtractor`.

Inputs are nonempty, rectangular `int64` token ID tensors with shape `[batch, sequence]`; padding is not supported. Each activation is a CPU tensor of shape `[batch × sequence, hidden_width]`, with sequences laid out in order. The extractor does not generate tokens.

`HookSite(layer, kind)` uses a zero-based layer index. Supported kinds are:

| Kind | Captured value |
|---|---|
| `resid_pre`, `resid_post` | Residual stream entering or leaving the layer |
| `mamba_out`, `moe_out`, `attn_out` | Layer mixer output, where that mixer exists |
| `attn_out_prelinear` | Attention output before its output projection |

Both extractors support `extract_async()` and `stream_to_disk()`. Disk streaming yields saved-shard receipts and keeps at most two batches in flight. `HFExtractor.extract_device()` returns CUDA activations for live training.

## Measured performance and agreement

In the recorded H200 run, the package's BF16 vLLM path saved activations at **5,535 input tokens/s** versus **3,930** for Transformers at batch size 64 (**1.41×**). A sibling fill-writer implementation reached **24,079 tokens/s** (**6.13×**). These figures cover 131,072 English tokens across 18 sites and describe package version 1.0.0; they are not a measurement of version 1.1.0. Each vLLM path passed the report's strict numerical agreement gates at only 3 of 18 sites. See the one-page [speed and agreement report](docs/report.pdf), its [source](docs/report.tex), and the [timing receipts](docs/extraction_speed_results.json) for scope and limitations.

## Repository guide

- `src/nemotron_extraction/` — public extractors, model hooks, vLLM worker and shard transport.
- `benchmarks/` — six standalone validation, throughput and attribution tools; see the [benchmark guide](benchmarks/README.md).
- `docs/` — report, measured speed receipts and controlled numerical experiments.
- `tests/` — package, benchmark and smoke tests.

## Development

From a checkout, install development dependencies and run the checks:

```bash
uv sync --group dev
uv run pytest
uvx ruff check src benchmarks tests
uvx ruff format --check src benchmarks tests
uv build
```

The report source can be compiled with `pdflatex -output-directory=docs docs/report.tex`.

## Citation and licence

See [CITATION.bib](CITATION.bib) for citation details. The package is MIT licensed; see [LICENSE](LICENSE). Model weights and datasets have their own licences.
