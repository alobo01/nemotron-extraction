# Benchmark guide

The scripts in this directory freeze inputs, run extractor workloads and analyse saved receipts. Model runs need the matching Transformers or vLLM environment and a suitable GPU; planning and analysis stages can run on CPU. From the repository root, install the required extras:

```bash
uv sync --extra hf --extra vllm --extra benchmark
```

Run `--help` on a script for all options. Outputs are written to the requested `--output` or run root; keep each experiment in a new directory so its receipts and source hashes remain traceable.

## Entry points

| Script | Use |
|---|---|
| `validate.py` | Compare Transformers and vLLM on frozen English, German and Spanish inputs. `all` prepares inputs, runs both backends and writes a report. |
| `sweep.py` | Measure extraction throughput and agreement across batch sizes 8–128. Run the Transformers backend before vLLM so the saved reference is available. |
| `extraction_speed.py` | Compare Transformers, two package configurations and two nemotron-sae sibling methods. Run each method separately, then compare the receipts. |
| `diagnose_divergence.py` | Trace backend outputs through model operations to locate and replay the first disagreement. |
| `precision_validation.py` | Compare layer-1 expert-operation traces with the checkpoint's Transformers source. |
| `moe_attribution.py` | Run controlled BF16 and FP32 MoE operator comparisons and expert-selection interventions. See [`../docs/moe_validation.md`](../docs/moe_validation.md) for the validated setup and findings. |

## Common workflows

Prepare the frozen token inputs once, then run all three languages sequentially:

```bash
uv run benchmarks/validate.py all --work outputs/tokens --output outputs/validation
```

For a shorter validation run, use `uv run benchmarks/validate.py run --work outputs/tokens --output outputs/pilot/en-0 --language en --tokens 16384`. For split cluster jobs, use `plan` to print ranges and `aggregate` to combine completed ranges. The planning stage submits no jobs.

To measure batch-size throughput, first run Transformers and then vLLM for each size, followed by the report stage:

```bash
uv run benchmarks/sweep.py backend --backend hf --batch-size 8 --work outputs/tokens --output outputs/sweep
uv run benchmarks/sweep.py backend --backend vllm --batch-size 8 --work outputs/tokens --output outputs/sweep
uv run benchmarks/sweep.py report --output outputs/sweep
```

The recorded speed study additionally requires a local nemotron-sae snapshot containing its `worker.py` and `attention.py`. Run all five methods in separate processes with the same work inputs and reference tree, then compare their receipts:

```bash
for method in transformers package_baseline package_optimized sibling_files sibling_fills; do
  uv run benchmarks/extraction_speed.py run --method "$method" --batch-size 8 --work outputs/tokens --reference path/to/nemotron-sae --output outputs/speed
done
uv run benchmarks/extraction_speed.py compare --work outputs/tokens --reference path/to/nemotron-sae --output outputs/speed
```

The supported methods are `transformers`, `package_baseline`, `package_optimized`, `sibling_files` and `sibling_fills`; batch sizes are 8 and 64. The report in [`../docs/report.pdf`](../docs/report.pdf) records the exact workload, source hashes and timing caveats.

For divergence analysis, run `inspect`, collect one trace per backend with `backend`, then use `analyse`. The `moe-replay` stage isolates layer 1. `precision_validation.py` consumes those traces plus the pinned model source and both source checksums; see `--help` for the required paths and hashes.

The MoE attribution harness uses a separate run root and staged reference source, traces and checkpoint weights. Follow its command sequence in [`../docs/moe_validation.md`](../docs/moe_validation.md); its verification stage checks the saved receipts and tensors without rerunning the experiment.

## Environment settings

| Variable | Used by | Effect |
|---|---|---|
| `NH_SINK_DIR` | `validate.py`, `sweep.py` | Place temporary vLLM shards on the selected filesystem. |
| `NH_GPU_MEMORY_UTILIZATION` | `validate.py` | Set the vLLM GPU memory fraction (default `0.96`). |
| `NH_KV_CACHE_BYTES` | `sweep.py`, `extraction_speed.py` | Set the vLLM KV-cache budget; the sweep default is 1,536 MiB, while the speed harness lets vLLM size it. |
| `NH_ENABLE_CHUNKED_PREFILL` | `extraction_speed.py` | Enable or disable chunked prefill (default enabled). |
| `NH_RUN_STORAGE_LIMIT_BYTES` | `extraction_speed.py` | Stop before a run exceeds its storage budget. |

## Shared benchmark code

`benchlib/` contains input and provenance helpers, agreement metrics, instrumentation, trace and worker support, and the sibling implementation. vLLM imports worker classes by module name inside its worker processes, so keep `benchmarks/` on `PYTHONPATH` when launching a custom worker environment.
