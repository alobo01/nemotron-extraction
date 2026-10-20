# Benchmarks

Each script is a standalone entry point. Install the package with the `benchmark` extra
(`uv pip install 'nemotron-extraction[hf,vllm,benchmark]==1.1.0'`), then run the scripts from the
repository root. They write JSON receipts and Markdown reports into `--output`. Stages that run a model need
a GPU; report, analysis and planning stages run on the CPU. All of them read the frozen inputs that
`validate.py prepare` writes to `--work`.

| Script | Question it answers |
|---|---|
| `validate.py` | Do the two backends produce the same activations? (the README results) |
| `sweep.py` | How do disk throughput and agreement change with batch size (8 to 128)? |
| `extraction_speed.py` | How fast is this package compared with Transformers and the nemotron-sae extraction methods? |
| `diagnose_divergence.py` | At which operation do the two backends first diverge? |
| `precision_validation.py` | Does vLLM's layer-1 expert operation match the checkpoint's Transformers source? |

## validate.py

```bash
uv run benchmarks/validate.py all --work outputs/tokens --output outputs/validation
```

`all` runs `prepare` once (pin revisions, download weights, tokenize 1,000,000 tokens per language), then
`run` for each language. `run` executes the `hf` and `vllm` backends in separate processes and writes a
report. When a language does not fit in one cluster job:

```bash
uv run benchmarks/validate.py run --work outputs/tokens --output pilot/en-0 --language en --tokens 16384
# pilot de-0 and es-0 the same way
uv run benchmarks/validate.py plan --pilot pilot                 # prints job ranges
uv run benchmarks/validate.py run ... --start S --tokens N       # one job per range
uv run benchmarks/validate.py aggregate --ranges r0 r1 ... --output outputs/validation/en
```

## sweep.py

```bash
uv run benchmarks/sweep.py backend --backend hf --batch-size 8 --work outputs/tokens --output outputs/sweep
uv run benchmarks/sweep.py backend --backend vllm --batch-size 8 --work outputs/tokens --output outputs/sweep
uv run benchmarks/sweep.py report --output outputs/sweep
```

Run `hf` before `vllm`, because the vLLM phase compares against the saved Transformers shards. Interrupted
phases resume from their last committed shard. `schedule` estimates job lengths from pilot runs.

## extraction_speed.py

```bash
uv run benchmarks/extraction_speed.py run --method transformers --batch-size 8 \
    --work outputs/tokens --reference path/to/nemotron-sae-snapshot --output outputs/speed
uv run benchmarks/extraction_speed.py compare --work outputs/tokens --reference ... --output outputs/speed
```

The methods are `transformers`, `package_baseline`, `package_optimized`, `sibling_files` and
`sibling_fills`, at batch sizes 8 and 64. Run each method in its own process; the batch-8 `transformers` run
is the numerical reference. `--reference` points to copies of nemotron-sae's `worker.py` and `attention.py`.

## diagnose_divergence.py

```bash
uv run benchmarks/diagnose_divergence.py inspect --work outputs/tokens --reference ... --output outputs/diagnosis
uv run benchmarks/diagnose_divergence.py backend --backend transformers --work outputs/tokens --reference ... --output outputs/diagnosis
uv run benchmarks/diagnose_divergence.py backend --backend vllm --work outputs/tokens --reference ... --output outputs/diagnosis
uv run benchmarks/diagnose_divergence.py analyse --output outputs/diagnosis
```

`analyse` names the first failing boundary and selects an operation to replay. Rerun both backends with
`--replay-selection outputs/diagnosis/analysis.json` and analyse again. `--integrity` checks transfer and
persistence against same-forward snapshots. `moe-replay` isolates the layer-1 mixture of experts.

## precision_validation.py

```bash
uv run benchmarks/precision_validation.py --work outputs/tokens --output outputs/precision \
    --transformers-trace T.safetensors --vllm-trace V.safetensors \
    --reference-source modeling_nemotron_h.py --reference-source-sha256 ... --upstream-source-sha256 ...
```

The traces are anchor traces from `diagnose_divergence.py backend`.

## Environment settings

| Variable | Used by | Effect |
|---|---|---|
| `NH_SINK_DIR` | validate, sweep | Directory for vLLM's temporary shards |
| `NH_GPU_MEMORY_UTILIZATION` | validate | vLLM memory fraction (default 0.96) |
| `NH_KV_CACHE_BYTES` | sweep, extraction_speed | vLLM KV-cache size; sweep defaults to 1536 MiB, extraction_speed lets vLLM size it |
| `NH_ENABLE_CHUNKED_PREFILL` | extraction_speed | Set `0` to disable chunked prefill (default on) |
| `NH_RUN_STORAGE_LIMIT_BYTES` | extraction_speed | Abort before writes would exceed this budget |

## benchlib/

Shared code, imported by module name because the scripts run from this directory:

- `common`: pinned inputs and provenance.
- `stats`: agreement metrics.
- `instrumentation`: GPU sampling.
- `sibling`: the nemotron-sae driver.
- `trace`, `workers`, `moe`, `precision`, `divergence`: diagnostics.

vLLM loads `benchlib.workers` and `benchlib.sibling` worker classes by name inside its worker processes.
