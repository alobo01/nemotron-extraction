# /// script
# requires-python = ">=3.10"
# dependencies = ["nemotron-extraction[vllm,benchmark]==1.0.0"]
# ///
"""Same-runtime layer-1 operator validation; full anchor is gated by a short probe.

Install the frozen wheel with uv. Infrastructure settings are supplied externally.
This compares the checkpoint's Transformers MoE source with vLLM's expert
operation. It does not claim full-model or extraction-throughput equivalence.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
from pathlib import Path
import sys
import time
import traceback

import torch
from safetensors import safe_open

from nemotron_extraction.diagnostics import digest, write_json
from nemotron_extraction.precision import ieee_precision
from targeted_validation import Utilisation
from diagnose_divergence import ANCHOR_SHA256, inputs, software


def gate_passed(result):
    return (result['strict_pass'] and result['exact_router_logits'] and
            result['expert_sets_equal'] and result['persisted_samples_exact'])


async def run(args):
    from vllm import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM
    manifest, rows = inputs(args.work)
    anchor_hash = hashlib.sha256(rows[0].cpu().numpy().astype('<i8').tobytes()).hexdigest()
    if anchor_hash != ANCHOR_SHA256:
        raise ValueError('Frozen anchor checksum differs')
    for trace in (args.transformers_trace, args.vllm_trace):
        with safe_open(str(trace), framework='pt') as file:
            if not torch.equal(file.get_tensor('token_ids'), rows[0]):
                raise ValueError('Input trace does not identify the complete frozen anchor')
    args.output.mkdir(parents=True, exist_ok=False)
    receipt = dict(status='running', scope='Layer-1 expert operation on frozen anchor inputs',
        full_model_equivalence_established=False, anchor_sha256=anchor_hash,
        model_revision=manifest['model_revision'], software=software(), command=sys.argv,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        input_trace_sha256={str(p): digest(p)
                            for p in (args.transformers_trace, args.vllm_trace)},
        reference_source_sha256=args.reference_source_sha256,
        upstream_model_source_sha256=args.upstream_source_sha256, cases={})
    write_json(args.output / 'result.json', receipt)
    config = dict(model=manifest['model_path'], dtype='bfloat16', trust_remote_code=True,
        tensor_parallel_size=1, enforce_eager=True, enable_prefix_caching=False,
        enable_chunked_prefill=True, max_model_len=2049, max_num_seqs=1,
        max_num_batched_tokens=2049, skip_tokenizer_init=True, gpu_memory_utilization=.96,
        mamba_cache_dtype='bfloat16', mamba_ssm_cache_dtype='float32',
        worker_extension_cls='nemotron_extraction.precision.PrecisionWorker')
    engine = None
    begun = time.perf_counter()
    try:
        with ieee_precision() as driver_precision, Utilisation(args.output / 'utilisation.jsonl') as monitor:
            receipt['driver_precision'] = driver_precision
            startup = time.perf_counter()
            engine = AsyncLLM.from_engine_args(AsyncEngineArgs(**config))
            receipt.update(startup_seconds=time.perf_counter()-startup, engine_configuration=config)
            monitor.phase = 'checkpoint_verification'
            started = time.perf_counter()
            receipt['preparation'] = (await engine.collective_rpc('precision_prepare', args=(
                manifest['model_path'], str(args.reference_source), args.reference_source_sha256)))[0]
            receipt['checkpoint_verification_seconds'] = time.perf_counter() - started
            for name, length, duplicate in [('probe128',128,False), ('anchor2048',2048,False),
                                             ('short37',37,False), ('duplicate_reversed',2048,True)]:
                monitor.phase = name
                result = (await engine.collective_rpc('precision_case', args=(
                    str(args.transformers_trace), str(args.vllm_trace), length, name,
                    str(args.output), duplicate)))[0]
                if result['torch'] != receipt['preparation']['torch'] or result['cuda'] != receipt['preparation']['cuda']:
                    raise ValueError('Replay runtime changed between stages')
                if result['precision'] != driver_precision:
                    raise ValueError('Worker precision differs from the driver')
                receipt['cases'][name] = result
                write_json(args.output / 'result.json', receipt)
                if name == 'probe128' and not gate_passed(result):
                    receipt.update(status='probe_failed', final_validation='not_run')
                    break
            else:
                receipt.update(status='passed' if all(gate_passed(r) for r in receipt['cases'].values()) else 'validation_failed',
                               final_validation='complete')
        receipt['utilisation'] = monitor.summary()
    except BaseException as error:
        receipt.update(status='error', error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        if engine is not None:
            try:
                receipt['cleanup_complete'] = (await engine.collective_rpc('precision_close'))[0]
            except Exception as error:
                receipt.update(cleanup_complete=False, cleanup_error=str(error))
            finally:
                engine.shutdown()
        receipt['wall_seconds'] = time.perf_counter() - begun
        write_json(args.output / 'result.json', receipt)
        lines = ['# Aligned precision: frozen layer-1 expert validation', '',
            f"Status: **{receipt['status']}**. Scope: the layer-1 expert operation; full-model equivalence is not established.", '',
            '| Case / input | Cosine mean ± std | Normalized MSE mean ± std | Worst 1% mean cosine | Maximum 1 − cosine | Maximum absolute error | Strict gates |',
            '|---|---:|---:|---:|---:|---:|---|']
        for name, result in receipt['cases'].items():
            for label, metric in result['comparisons'].items():
                lines.append(f"| {name} / {label} | {metric['mean_cosine']:.8f} ± {metric['cosine_std']:.3g} | {metric['normalized_mse']:.8g} ± {metric['normalized_mse_std']:.3g} | {metric['worst_one_percent']['mean_cosine']:.8f} | {metric['maximum_cosine_disagreement']:.8g} | {metric['maximum_absolute_error']:.8g} | {'pass' if metric['passed'] else 'fail'} |")
        if 'error' in receipt:
            lines.extend(['', 'Execution error: ' + receipt['error']])
        lines.extend(['', 'Operator timings include expert calculations only and are not activation-extraction speedups.'])
        lines.extend(['', 'The worst 1% uses ceil(0.01 × token rows), ranked separately by cosine disagreement and squared error. JSON retains both row lists and their error share. Normalized MSE divides squared row error by mean reference squared row norm, so its mean equals total squared error divided by total reference squared magnitude.'])
        (args.output / 'report.md').write_text('\n'.join(lines)+'\n')
    return 0 if receipt['status'] == 'passed' and receipt.get('cleanup_complete') else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--transformers-trace', type=Path, required=True)
    parser.add_argument('--vllm-trace', type=Path, required=True)
    parser.add_argument('--reference-source', type=Path, required=True)
    parser.add_argument('--reference-source-sha256', required=True)
    parser.add_argument('--upstream-source-sha256', required=True,
                        help='SHA-256 of the original upstream model source before bundling')
    args = parser.parse_args()
    torch.set_num_threads(4)
    return asyncio.run(run(args))


if __name__ == '__main__':
    raise SystemExit(main())
