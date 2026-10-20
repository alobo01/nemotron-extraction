# /// script
# requires-python = ">=3.10"
# dependencies = ["nemotron-extraction[hf,vllm,benchmark]==1.1.0"]
# ///
"""Locate where the Transformers and vLLM activations first diverge.

Stages: inspect, backend (per engine, optionally --integrity), analyse, then
backend --replay-selection and analyse again; moe-replay isolates layer-1 experts.
Diagnostics synchronize the GPU, so their timings are not extraction throughput.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import traceback
from pathlib import Path

import torch
from benchlib.common import (
    ANCHOR_SHA256,
    MODEL,
    MODEL_REVISION,
    anchor_digest,
    anchor_inputs,
    digest,
    is_memory_failure,
    package_hashes,
    versions,
    write_json,
)
from benchlib.divergence import analyse, write_moe_replay_report
from benchlib.instrumentation import Utilisation
from benchlib.trace import TraceHooks, runtime_config
from safetensors.torch import load_file, save_file

from nemotron_extraction import HFExtractor, HookSite, VLLMExtractor
from nemotron_extraction.hooks import repair_time_step_limits
from nemotron_extraction.transport import SLOTS, atomic_path

# Two duplicated and two reordered prompts, truncated to keep the integrity case small.
INTEGRITY_ORDER = [2, 0, 2, 1]


def cases(rows, pilot=False, names=None):
    selected = [("short128", rows[:1, :128].contiguous(), 0), ("alone", rows[:1], 0)]
    if not pilot:
        for size in (8, 64):
            selected += [(f"b{size}", rows[:size], 0), (f"b{size}_reversed", rows[:size].flip(0), size - 1)]
        selected += [("duplicates", rows[[0, 2, 0, 1]], 0), ("short37", rows[:1, :37].contiguous(), 0)]
    if names:
        unknown = set(names) - {name for name, _, _ in selected}
        if unknown:
            raise ValueError("Unknown diagnostic cases: " + str(sorted(unknown)))
        selected = [item for item in selected if item[0] in names]
    return selected


def selected_cases(args, rows, spec=None):
    names = getattr(args, "cases", None)
    if spec:
        names = [spec.get("case", "alone")]
    return cases(rows, args.pilot or bool(spec), names)


def inspect_inputs(args):
    manifest, _ = anchor_inputs(args.work)
    weights = Path(manifest["model_path"])
    index = weights / "model.safetensors.index.json"
    shards = sorted(set(json.loads(index.read_text())["weight_map"].values()))
    if not all((weights / name).is_file() and (weights / name).stat().st_size for name in shards):
        raise ValueError("Weight shards missing")
    record = dict(
        model=MODEL,
        model_revision=MODEL_REVISION,
        manifest_sha256=digest(args.work / "manifest.json"),
        english_tokens_sha256=manifest["languages"]["en"]["token_sha256"],
        script_sha256=digest(__file__),
        package_source_sha256=package_hashes(),
        sibling_source_sha256={p.name: digest(p) for p in sorted(args.reference.glob("*.py"))},
        model_config=json.loads((weights / "config.json").read_text()),
        weight_index_sha256=digest(index),
        weight_shards={name: (weights / name).stat().st_size for name in shards},
        command=sys.argv,
        software=versions(),
        gates=dict(mean_cosine=0.9999, minimum_cosine=0.9990, relative_l2=0.02),
        synchronization="Diagnostics intentionally synchronize; timings are not extraction throughput",
    )
    target = args.output / "provenance.json"
    if target.exists():
        previous = json.loads(target.read_text())
        for key in (
            "model_revision",
            "manifest_sha256",
            "script_sha256",
            "package_source_sha256",
            "sibling_source_sha256",
        ):
            if record[key] != previous[key]:
                raise ValueError("Provenance changed: " + key)
    else:
        write_json(target, record)
    return record


def replay_spec(args):
    if not args.replay_selection:
        return None
    result = json.loads(args.replay_selection.read_text())["replay"]
    if result is None:
        raise ValueError("No operation selected for replay")
    return result


def transformers_backend(args, rows, manifest, destination, monitor):
    from transformers import AutoModelForCausalLM

    started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        manifest["model_path"],
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
        local_files_only=True,
    )
    repair_time_step_limits(model)
    model.eval()
    result = dict(startup_seconds=time.perf_counter() - started, runtime=runtime_config(model), cases={})
    decoder = getattr(model, "model", model)
    trace = TraceHooks(model)
    spec = replay_spec(args)
    if spec:
        trace.replay = dict(
            layer=spec["layer"], operation=spec["operation"], input=load_file(spec["path"])[spec["key"]]
        )
        trace.details(spec["layer"])
    try:
        for name, tokens, anchor in selected_cases(args, rows, spec):
            path = destination / (name + ".safetensors")
            receipt = destination / (name + ".json")
            if receipt.exists():
                r = json.loads(receipt.read_text())
                result["cases"][name] = r
                if r["status"] == "complete" and not path.is_file():
                    raise ValueError("Missing completed trace")
                continue
            monitor.phase = name
            trace.reset()
            trace.span = (anchor * tokens.shape[1], (anchor + 1) * tokens.shape[1])
            begun = time.perf_counter()
            record = dict(status="running", batch_size=len(tokens), sequence_length=tokens.shape[1], anchor=anchor)
            try:
                with torch.inference_mode():
                    decoder(input_ids=tokens.to("cuda"), use_cache=False)
                torch.cuda.synchronize()
                values = {k: v.cpu().contiguous() for k, v in trace.values.items()}
                values["token_ids"] = tokens[anchor].clone()
                if not values or "embedding" not in values:
                    raise ValueError("Missing embedding trace")
                if not all(bool(torch.isfinite(v).all()) for v in values.values()):
                    raise ValueError("Nonfinite trace")
                with atomic_path(path) as partial_path:
                    save_file(values, str(partial_path))
                restored = load_file(str(path))
                if not all(torch.equal(v, restored[k]) for k, v in values.items()):
                    raise ValueError("Trace persistence mismatch")
                record.update(
                    status="complete",
                    path=str(path),
                    topology=trace.topology,
                    trace_persistence_exact=True,
                    actual_forward_batches=[len(tokens)],
                    operator_observability=trace.detail_observability,
                )
            except torch.OutOfMemoryError as error:
                record.update(status="memory_failure", error=str(error))
                torch.cuda.empty_cache()
            record["diagnostic_wall_seconds"] = time.perf_counter() - begun
            record["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            write_json(receipt, record)
            result["cases"][name] = record
    finally:
        trace.close()
    return result


async def vllm_backend(args, rows, manifest, destination, monitor):
    from vllm import AsyncEngineArgs, SamplingParams
    from vllm.v1.engine.async_llm import AsyncLLM

    started = time.perf_counter()
    config = dict(
        model=manifest["model_path"],
        dtype="bfloat16",
        trust_remote_code=True,
        tensor_parallel_size=1,
        enforce_eager=True,
        enable_prefix_caching=False,
        enable_chunked_prefill=not args.no_chunked_prefill,
        max_model_len=2049,
        max_num_seqs=64,
        max_num_batched_tokens=64 * 2049,
        skip_tokenizer_init=True,
        gpu_memory_utilization=0.96,
        worker_extension_cls="benchlib.workers.DiagnosticWorker",
    )
    engine = AsyncLLM.from_engine_args(AsyncEngineArgs(**config))
    result = dict(startup_seconds=time.perf_counter() - started, engine_configuration=config, cases={})
    installed = False
    try:
        install_start = time.perf_counter()
        result["runtime"] = (await engine.collective_rpc("diag_install", args=(str(args.reference),)))[0]
        installed = True
        result["hook_install_seconds"] = time.perf_counter() - install_start
        spec = replay_spec(args)
        requests = selected_cases(args, rows, spec)
        if args.integrity:
            requests = [
                ("duplicate_reordered", rows[INTEGRITY_ORDER, :128].contiguous(), 1),
                ("short37", rows[:1, :37].contiguous(), 0),
            ]
        for name, tokens, anchor in requests:
            path = destination / (name + ".safetensors")
            receipt = destination / (name + ".json")
            if receipt.exists():
                r = json.loads(receipt.read_text())
                result["cases"][name] = r
                if r["status"] == "complete" and not path.is_file():
                    raise ValueError("Missing completed trace")
                continue
            monitor.phase = name
            ids = [f"diagnostic-{name}-{i}" for i in range(len(tokens))]
            begun = time.perf_counter()
            await engine.collective_rpc(
                "diag_begin", args=(name, ids, tokens.tolist(), anchor, str(path), args.integrity, spec)
            )

            async def request(index):
                async for _ in engine.generate(
                    {"prompt_token_ids": tokens[index].tolist()},
                    SamplingParams(max_tokens=1, temperature=0),
                    ids[index],
                ):
                    pass

            # Admit in reverse; the scheduler receipts record the order actually executed.
            await asyncio.gather(*(request(i) for i in reversed(range(len(tokens)))))
            record = (await engine.collective_rpc("diag_finish"))[0]
            record.update(
                status="complete",
                batch_size=len(tokens),
                sequence_length=tokens.shape[1],
                anchor=anchor,
                diagnostic_wall_seconds=time.perf_counter() - begun,
            )
            write_json(receipt, record)
            result["cases"][name] = record
    finally:
        try:
            if installed:
                await engine.collective_rpc("diag_close")
        finally:
            engine.shutdown()
    return result


async def moe_replay(args):
    from vllm import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    manifest, rows = anchor_inputs(args.work)
    anchor_hash = anchor_digest(rows)
    if anchor_hash != ANCHOR_SHA256:
        raise ValueError(f"Frozen anchor checksum differs: {anchor_hash}")
    for trace in (args.transformers_trace, args.vllm_trace):
        if not trace.is_file():
            raise FileNotFoundError(trace)
    args.output.mkdir(parents=True, exist_ok=True)
    engine_config = dict(
        model=manifest["model_path"],
        dtype="bfloat16",
        trust_remote_code=True,
        tensor_parallel_size=1,
        enforce_eager=True,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        max_model_len=2049,
        max_num_seqs=1,
        max_num_batched_tokens=2049,
        skip_tokenizer_init=True,
        gpu_memory_utilization=0.96,
        worker_extension_cls="benchlib.workers.DiagnosticWorker",
    )
    engine = AsyncLLM.from_engine_args(AsyncEngineArgs(**engine_config))
    output = args.output / "layer1-moe-replay.safetensors"
    try:
        worker = (
            await engine.collective_rpc(
                "diag_moe_replay", args=(str(args.transformers_trace), str(args.vllm_trace), str(output), 1)
            )
        )[0]
    finally:
        engine.shutdown()
    tensors = load_file(str(output))
    if not torch.equal(tensors["token_ids"], rows[0]):
        raise ValueError("Diagnostic traces do not identify frozen anchor sequence 0")
    source = {
        "model": manifest["model"],
        "model_revision": manifest["model_revision"],
        "tokenizer_revision": manifest["tokenizer_revision"],
        "token_file_sha256": manifest["languages"]["en"]["token_sha256"],
        "anchor_token_ids_i64le_sha256": anchor_hash,
        "package_script_sha256": digest(__file__),
    }
    return write_moe_replay_report(args.output, tensors, worker, engine_config, source, output)


class IntegrityExtractor(VLLMExtractor):
    _worker_extension_cls = "benchlib.workers.IntegrityWorker"


class IntegrityHFExtractor(HFExtractor):
    """Same-forward oracle around the unchanged Transformers persistence path."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.oracles = {}
        decoder = getattr(self.model, "model", self.model)

        def oracle(module, args, output):
            self.oracles[self.diag_slot] = {
                k: v.detach().reshape(-1, v.shape[-1]).clone().cpu() for k, v in self.captured.items()
            }

        self.handles.append(decoder.register_forward_hook(oracle))

    def _capture(self, tokens, slot):
        self.diag_slot = slot
        return super()._capture(tokens, slot)

    def _publish(self, tokens, slot, capture, path, index):
        oracle = self.oracles.pop(slot)
        occupied = SLOTS - len(self.pool.free)
        time.sleep(0.5)  # hold this slot so the next capture fills the other buffer
        result = super()._publish(tokens, slot, capture, path, index)
        saved = load_file(result["path"])
        exact = torch.equal(saved["token_ids"], tokens) and all(torch.equal(saved[k], v) for k, v in oracle.items())
        if not exact:
            raise RuntimeError("Transformers asynchronous transfer/persistence mismatch")
        result.update(diagnostic_transport_exact=True, diagnostic_slot=slot, diagnostic_occupied_slots=occupied)
        return result


def package_integrity(args, rows, manifest, destination, monitor):
    site_path = args.output / "sites.json"
    if not site_path.exists():
        raise ValueError("Run Transformers backend first to resolve integrity sites")
    sites = [HookSite(**s) for s in json.loads(site_path.read_text())]
    started = time.perf_counter()
    extractor = (
        IntegrityHFExtractor(manifest["model_path"], sites, local_files_only=True)
        if args.integrity == "transformers"
        else IntegrityExtractor(
            manifest["model_path"],
            sites,
            max_model_len=2049,
            sequences_per_call=4,
            enable_chunked_prefill=True,
            gpu_memory_utilization=0.96,
            sink_dir=str(destination),
        )
    )
    with extractor:
        result = dict(startup_seconds=time.perf_counter() - started)
        monitor.phase = "two_pending_batches"
        batch = rows[INTEGRITY_ORDER, :128].contiguous()

        async def collect():
            return [
                r
                async for r in extractor.stream_to_disk(
                    [batch, batch.flip(0), rows[:1, :37].contiguous()], destination / "shards"
                )
            ]

        receipts = asyncio.run(collect())
        if not all(r["diagnostic_transport_exact"] for r in receipts):
            raise ValueError("Transfer mismatch")
        occupied = max(r["diagnostic_occupied_slots"] for r in receipts)
        if occupied != SLOTS:
            raise ValueError("Both pending buffer slots were not exercised")
        result.update(
            receipts=receipts, two_slots_observed=True, actual_forward_batches=extractor.actual_forward_batches
        )
    return result


def backend(args):
    torch.set_num_threads(4)
    torch.manual_seed(0)
    inspect_inputs(args)
    manifest, rows = anchor_inputs(args.work)
    variant = args.label or args.backend
    if args.integrity:
        variant += "-" + args.integrity
    if args.no_chunked_prefill:
        variant += "-no-chunked"
    if args.replay_selection:
        variant += "-replay"
    if args.pilot:
        variant += "-pilot"
    destination = args.output / variant
    destination.mkdir(parents=True, exist_ok=True)
    receipt = destination / "result.json"
    if receipt.exists():
        previous = json.loads(receipt.read_text())
        if previous["status"] == "complete":
            print("Preserving complete stage", variant, flush=True)
            return 0
        raise FileExistsError("Preserve failed result; use a new output directory")
    record = dict(
        status="running",
        backend=args.backend,
        software=versions(),
        command=sys.argv,
        provenance_sha256=digest(args.output / "provenance.json"),
        gpu=torch.cuda.get_device_name(0),
    )
    monitor = Utilisation(destination / "utilisation.jsonl")
    begun = time.perf_counter()
    try:
        with monitor:
            if args.integrity in ("package", "transformers"):
                result = package_integrity(args, rows, manifest, destination, monitor)
            elif args.backend == "transformers":
                result = transformers_backend(args, rows, manifest, destination, monitor)
                # The same pinned topology also defines the 18 original integrity sites.
                config = json.loads((Path(manifest["model_path"]) / "config.json").read_text())
                kinds = {"M": "mamba_out", "*": "attn_out", "E": "moe_out"}
                pattern = config["hybrid_override_pattern"]
                sites = []
                for kind in ("resid_pre", "resid_post", "mamba_out", "moe_out", "attn_out", "attn_out_prelinear"):
                    eligible = [
                        i
                        for i, c in enumerate(pattern)
                        if kind.startswith("resid_")
                        or kinds.get(c) == ("attn_out" if kind == "attn_out_prelinear" else kind)
                    ]
                    if not eligible:
                        raise ValueError("Unknown checkpoint topology for " + kind)
                    sites += [
                        dict(layer=i, kind=kind)
                        for i in sorted({eligible[0], eligible[(len(eligible) - 1) // 2], eligible[-1]})
                    ]
                write_json(args.output / "sites.json", sites)
            else:
                result = asyncio.run(vllm_backend(args, rows, manifest, destination, monitor))
            record.update(result, status="complete")
    except Exception as error:
        record.update(
            status="memory_failure" if is_memory_failure(error) else "error",
            error_type=type(error).__name__,
            error=str(error),
            traceback=traceback.format_exc(),
        )
    record.update(utilisation=monitor.summary(), process_wall_seconds=time.perf_counter() - begun)
    write_json(receipt, record)
    print(
        json.dumps({k: record.get(k) for k in ("status", "error", "startup_seconds", "process_wall_seconds")}),
        flush=True,
    )
    return 0 if record["status"] == "complete" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    for name in ("inspect", "backend", "analyse", "moe-replay"):
        p = sub.add_parser(name)
        p.add_argument("--output", type=Path, required=True)
        if name in ("inspect", "backend"):
            p.add_argument("--work", type=Path, required=True)
            p.add_argument("--reference", type=Path, required=True)
        if name == "backend":
            p.add_argument("--backend", choices=("transformers", "vllm"), required=True)
            p.add_argument("--label", help="Separate receipt directory for an explicit runtime attempt")
            p.add_argument("--pilot", action="store_true")
            p.add_argument("--no-chunked-prefill", action="store_true")
            p.add_argument("--cases", nargs="+", help="Run only named cases; preserves full requested batch sizes")
            p.add_argument("--integrity", choices=("package", "transformers", "sibling_files", "sibling_fills"))
            p.add_argument("--replay-selection", type=Path)
        if name == "moe-replay":
            p.add_argument("--work", type=Path, required=True)
            p.add_argument("--transformers-trace", type=Path, required=True)
            p.add_argument("--vllm-trace", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.stage == "moe-replay":
        return asyncio.run(moe_replay(args))
    return {"inspect": inspect_inputs, "backend": backend, "analyse": analyse}[args.stage](args)


if __name__ == "__main__":
    result = main()
    raise SystemExit(result if isinstance(result, int) else 0)
