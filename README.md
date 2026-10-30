# Nemotron extraction

Capture token-aligned activations from NVIDIA's [Nemotron 3 Nano 30B A3B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16) with either Hugging Face Transformers (PyTorch) or vLLM. Both backends share one interface: you choose model sites, pass token IDs, and get one activation row per input token, in memory or as files on disk. The package was built to feed sparse-autoencoder training, where tokens per second and exact row alignment both matter.

## Install

Python 3.10 or later. vLLM needs Linux and an NVIDIA GPU; the PyTorch backend also runs on a CPU.

```bash
uv venv
uv pip install 'nemotron-extraction[hf,vllm] @ git+https://github.com/alobo01/nemotron-extraction'
```

Use `[hf]` or `[vllm]` for one backend. The PyPI release, 1.1.0, predates chunked prefill, concurrent waves and the precision modes.

## Try it without a GPU

[`examples/toy_extraction.py`](examples/toy_extraction.py) runs the real PyTorch extractor on a three-layer stand-in with Mamba, MoE and attention mixers, so it needs neither the checkpoint nor a GPU:

```bash
python examples/toy_extraction.py outputs/toy
```

```text
L0_mamba_out             (10, 8)
L1_moe_out               (10, 8)
L2_attn_out_prelinear    (10, 8)
L2_resid_post            (10, 8)
wave 0: outputs/toy/000000 (5 files, rows (10, 8))
wave 1: outputs/toy/000001 (5 files, rows (10, 8))
```

Two sequences of five tokens give ten rows per site. The streamed waves are directories of per-site files with a checksummed receipt, and their rows match the in-memory result.

## Use with Nemotron 3 Nano

```python
import torch
from nemotron_extraction import HookSite, VLLMExtractor

sites = [HookSite(12, "resid_post"), HookSite(12, "attn_out_prelinear")]
tokens = torch.randint(0, 1000, (8, 2048))  # [batch, sequence], int64, no padding

with VLLMExtractor("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", sites, sequences_per_call=8) as extractor:
    batch = extractor.extract(tokens)  # batch.activations["L12_resid_post"]: [8 * 2048, 2688] on the CPU
```

`HFExtractor` takes the same arguments and also accepts a loaded PyTorch module. Both offer `extract`, `extract_async` and `stream_to_disk`, which yields one receipt per saved wave, in order. `HFExtractor.extract_device` keeps the activations on the GPU for live training. Neither backend generates text.

**Sites.** `HookSite(layer, kind)` uses zero-based layers. Each mixer kind exists only on layers of that type.

| Kind | Captured value |
|---|---|
| `resid_pre`, `resid_post` | Residual stream entering or leaving the layer |
| `mamba_out`, `moe_out`, `attn_out` | Output of the layer's mixer |
| `attn_out_prelinear` | Attention output before its output projection |

**Options** (constructor arguments; vLLM only unless noted):

| Option | Effect |
|---|---|
| `sequences_per_call` | Prompts per capture wave; `"auto"` sizes it from an `AutoBatchConfig` memory profile |
| `enable_chunked_prefill`, `max_num_batched_tokens` | Split long prompts across forward passes under a token budget |
| `concurrent_waves`, `capture_slots` | Let up to `capture_slots` waves share forward passes, so the GPU is not idle between waves |
| `publication_layout="per_site"`, `site_write_threads` | One file per site, written in parallel and published atomically with SHA-256 receipts (both backends) |
| `precision="matched"` | IEEE float32 matmuls and, on vLLM, a float32 MoE router, matching the PyTorch reference (both backends) |
| `set_active_sites`, `set_sequences_per_call` | Change sites or wave size between streams without reloading the model |

## How it works

**Sites become hooks.** `hooks.py` finds the decoder layers and their mixers, and registers a forward hook per site. In PyTorch the hooks run on the model directly. In vLLM they run inside the GPU worker process, through a worker extension, so activations never pass through the engine's output path.

**Rows are routed by request and offset.** vLLM flattens all scheduled prompts into one token batch per forward pass. Before each pass the worker reads the scheduler's metadata: request ID, tokens already computed and position in the flat batch. It then copies each request's slice to `row × length + offset` in that wave's output. This keeps rows exact under chunked prefill, interleaved requests and waves that share a pass. A span that arrives out of order, as after a pre-emption, fails the wave instead of being written.

**Copies and writes overlap the forward pass.** Captured tensors are copied to pinned host buffers on a separate CUDA stream. A wave is published only after all its spans have arrived. Writer threads then validate, checksum, sync and rename the files, while the GPU runs the next wave. The number of buffer slots bounds host memory.

## Repository layout

| Path | Contents |
|---|---|
| `src/nemotron_extraction/types.py` | `HookSite`, site kinds and the `ActivationBatch` result types |
| `src/nemotron_extraction/hooks.py` | Topology discovery and hook installation (residual, mixer, pre-projection) |
| `src/nemotron_extraction/hf.py` | `HFExtractor`: the PyTorch backend and reference |
| `src/nemotron_extraction/vllm.py` | `VLLMExtractor`: engine set-up, wave admission and the ordered receipt stream |
| `src/nemotron_extraction/worker.py` | vLLM worker extension: per-request spans, GPU-to-host copies and wave sealing |
| `src/nemotron_extraction/transport.py` | Pinned buffer pool and durable monolithic or per-site publication |
| `src/nemotron_extraction/precision.py` | Native and matched precision modes, including the float32 router |
| `src/nemotron_extraction/batching.py` | `AutoBatchConfig`: wave size from a memory profile |
| `examples/` | The CPU toy above |
| `benchmarks/` | Validation, speed and numerical-diagnosis tools; see the [benchmark guide](benchmarks/README.md) |
| `docs/` | Report and receipts of earlier speed and numerical studies |
| `tests/` | CPU tests with stand-in models and a fake chunking engine; no GPU needed |

## Validation and speed

On an H100, with 18 sites and per-site saving, vLLM ran the forward pass at 56,000–62,000 tokens/s for prompts of 1,024–16,384 tokens in waves of four or more prompts. The PyTorch backend reached 10,700–12,300. Writing to a network filesystem at about 5 GB/s, roughly 47,000 tokens/s for 18 sites, then caps end-to-end extraction at about 30,000 tokens/s for prompts of 4,096 tokens or more, against 7,000 for PyTorch. Concurrent waves produced the same activations as waves run one at a time with the same token budget.

Activations from the two backends differ by bfloat16 rounding, mostly from Mamba and MoE kernels that sum in a different order. Against a float32 PyTorch reference over 65,536 tokens each of English, German and Spanish, both backends are equally close at every site:

| Candidate | Relative L2, mean of 18 sites | Relative L2, worst site (layer 27 MoE output) | Mean cosine |
|---|---|---|---|
| PyTorch, bfloat16 | 0.037 | 0.104 | 0.999 |
| vLLM, bfloat16 | 0.037 | 0.105 | 0.999 |
| vLLM, bfloat16, 512-token chunked prefill | 0.037 | 0.105 | 0.999 |
| vLLM, float32 | 0.019 | 0.061 | 0.9997 |

vLLM in float32 is closer but not identical to PyTorch in float32, because its kernels still sum in a different order. `benchmarks/validate.py` and `benchmarks/scaling.py` reproduce these measurements.

## Development

```bash
uv sync --group dev
uv run pytest
uvx ruff check src benchmarks tests examples
uv build
```

## Citation and licence

See [CITATION.bib](CITATION.bib). The code is MIT licensed ([LICENSE](LICENSE)); model weights and datasets carry their own licences.
