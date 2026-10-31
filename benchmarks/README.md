# Benchmark guide

These scripts measure how fast the extractors run and how closely their activations agree. Model runs need a GPU and the matching Transformers or vLLM environment; planning and analysis stages run on a CPU. From the repository root:

```bash
uv sync --extra hf --extra vllm --extra benchmark
```

Every script has `--help`. Write each experiment to a new output directory: receipts record source hashes and settings, and completed receipts are never overwritten.

## Scripts

| Script | Question it answers |
|---|---|
| `validate.py` | Do two configurations produce the same activations on frozen English, German and Spanish tokens? |
| `scaling.py` | How do forward speed and saving speed change with prompt length and batch size? |
| `sweep.py` | How do throughput and agreement change with batch size, at 2,048-token prompts? |
| `extraction_speed.py` | How does the package compare with the nemotron-sae extraction methods? (historical study, see `../docs/report.pdf`) |
| `language_centroids.py` | How fast are per-language mean block outputs when nothing is saved? See the main README. |
| `diagnose_divergence.py` | At which operation do the two backends first disagree? |
| `precision_validation.py`, `moe_attribution.py` | Is a disagreement explained by Mamba or MoE arithmetic? See [`../docs/moe_validation.md`](../docs/moe_validation.md). |

## Validating agreement

`prepare` freezes tokens once. Each `backend` run then either writes a reference or compares itself with one:

```bash
uv run benchmarks/validate.py prepare --work outputs/tokens
uv run benchmarks/validate.py backend --backend hf --work outputs/tokens --language en --tokens 16384 \
    --output outputs/val/hf
uv run benchmarks/validate.py backend --backend vllm --work outputs/tokens --language en --tokens 16384 \
    --reference outputs/val/hf/reference --output outputs/val/vllm
```

A float32 PyTorch run is the ground truth for judging both backends. Use batch size 1, because the float32 PyTorch Mamba path needs far more memory per sequence. Offload part of the model if the GPU cannot hold its 126 GB of weights:

```bash
uv run benchmarks/validate.py backend --backend hf --dtype float32 --hf-device-map auto \
    --hf-max-gpu-memory 110GiB --work outputs/tokens --language en --tokens 65536 --batch-size 1 \
    --output outputs/val/hf-fp32
uv run benchmarks/validate.py backend --backend vllm --reference outputs/val/hf-fp32/reference \
    --work outputs/tokens --language en --tokens 65536 --batch-size 1 --output outputs/val/vllm
```

Options select what is compared:

| Option | Effect |
|---|---|
| `--dtype float32` | Run in float32. A float32 reference can judge bfloat16 runs. |
| `--hf-device-map auto --hf-max-gpu-memory 45GiB` | Offload part of the PyTorch model to the CPU, for a float32 reference that does not fit on the GPU |
| `--precision matched` | Matched precision mode (see the main README) |
| `--max-num-batched-tokens N`, `--concurrent-waves`, `--sequences-per-call N` | vLLM chunked prefill and concurrent waves |
| `--write-reference` | Make a vLLM run the reference, e.g. whole-prompt prefill against chunked prefill |

Each comparison reports, per site, mean, minimum and first-percentile cosine similarity, the mean cosine of the worst 1% of tokens, relative L2 error, normalised MSE (squared relative L2) with its worst-1% mean, and maximum absolute error. The worst-1% values exist only for runs that were not merged from ranges with `aggregate`. `all` runs the standard three-language PyTorch-against-vLLM comparison, and `plan` / `aggregate` split long runs into cluster-sized ranges.

## Measuring speed

`scaling.py run` measures one configuration (`hf`, `vllm-whole`, `vllm-chunked` or `vllm-continuous`) over a grid of prompt lengths and batch sizes. Each cell is two waves through the real `stream_to_disk` path. Forward time, GPU-to-host copy time and write time are reported separately from the receipts.

- **Saving cap:** cells whose wave exceeds `--saving-cap-bytes` capture on the GPU and skip the copy and write, so very large cells measure the forward pass only.
- **Storage:** written shards are deleted as soon as their receipt arrives.
- **PyTorch limit:** one PyTorch forward pass cannot exceed about 208,000 tokens (prompt length × batch size). From about 200,000 tokens an 80–94 GB GPU runs out of memory in the Mamba scan. A GPU with more memory crashes instead: above 208,423 tokens, `mamba_ssm`'s `_chunk_cumsum_fwd_kernel` computes its offset into the 10,304-wide `in_proj` output with 32-bit integers, and the offset passes 2³¹. Probes at prompt lengths of 2,048, 4,096 and 8,192 tokens all switched from out-of-memory to a crash between 206,848 and 208,896 tokens. `scaling.py` skips PyTorch cells above 208,000 tokens (`not run: PyTorch forward above its token limit`); vLLM cells are not skipped.

```bash
uv run benchmarks/scaling.py run --config vllm-continuous --work outputs/tokens --sites sites.json \
    --bytes-per-token 105216 --output outputs/scaling
```

## Language centroids

`language_centroids.py` reads a JSONL manifest (`language_nllb`, `example_id`, `text`) and writes `centroids.safetensors` and `runtime.json`. It first runs `--warmup` rows (default 512) and discards them, so `extraction_seconds` excludes first-use kernel compilation and tuning; `engine_startup_seconds` and `warmup_seconds` report those separately. Pass further vLLM engine arguments as JSON, for example `--engine-options '{"language_model_only": true}'` for a multimodal checkpoint.

On a shared node, check the CPU affinity of the vLLM workers (`taskset -cp PID`). On a 4-GPU job the two workers nearest the second CPU socket were pinned to 4 cores and spent over an hour in vLLM's CPU-side FP8 expert-weight shuffle, while the other two finished in minutes. `taskset -a -pc LIST PID` on the slow workers fixed it. `NCCL_IGNORE_CPU_AFFINITY=1` did not.

## Environment settings

| Variable | Used by | Effect |
|---|---|---|
| `NH_SINK_DIR` | `validate.py`, `sweep.py`, `scaling.py` | Directory for temporary vLLM shards |
| `NH_GPU_MEMORY_UTILIZATION` | `validate.py`, `scaling.py` | vLLM GPU memory fraction |
| `NH_KV_CACHE_BYTES` | `sweep.py`, `extraction_speed.py` | vLLM KV-cache budget |
| `NH_ENABLE_CHUNKED_PREFILL` | `extraction_speed.py` | Chunked prefill on or off (default on) |
| `NH_RUN_STORAGE_LIMIT_BYTES` | `extraction_speed.py` | Stop before a run exceeds its storage budget |

## Shared code

`benchlib/` holds input and provenance helpers, agreement statistics, instrumentation, trace and worker support. vLLM imports custom worker classes by module name inside its worker processes, so keep `benchmarks/` on `PYTHONPATH` when launching them.
