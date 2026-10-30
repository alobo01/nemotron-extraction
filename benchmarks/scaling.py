# /// script
# requires-python = ">=3.10"
# dependencies = ["nemotron-extraction[hf,vllm,benchmark]==1.1.0"]
# ///
"""Extraction speed over prompt length and batch size, forward and saving timed separately.

Run one configuration per process, then `report` over the result files:

    python benchmarks/scaling.py run --config vllm-continuous --work TOKENS --sites sites.json --output OUT
    python benchmarks/scaling.py report --results OUT/*.json --output OUT

Every cell runs two waves of B prompts of L tokens through the production `stream_to_disk`
path. Cells whose wave fits the saving cap also copy to pinned memory and publish per-site
shards (sha256, fsync, rename), which are deleted as soon as their receipt arrives. Larger
cells capture on the device and release the rows after each forward: forward speed only.
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import os
import shutil
import time
import traceback
from concurrent.futures import Future
from pathlib import Path

import torch
from benchlib.common import MODEL, is_memory_failure, load_tokens, package_hashes, versions, write_json

from nemotron_extraction import HFExtractor, HookSite, VLLMExtractor
from nemotron_extraction.precision import precision_settings
from nemotron_extraction.transport import MEMORY_KEYS, seconds
from nemotron_extraction.worker import ActivationHookExtension

LENGTHS = (1024, 4096, 16384, 65536, 100000)
BATCHES = (1, 4, 16, 64)
WAVE_TOKEN_CAP = 1 << 20
CONFIGS = {
    # name: backend, extractor options; whole prefill gets a budget of one longest prompt
    "hf": ("hf", {}),
    "vllm-whole": ("vllm", {}),
    "vllm-chunked": ("vllm", dict(enable_chunked_prefill=True, max_num_batched_tokens=8192)),
    "vllm-continuous": (
        "vllm",
        dict(enable_chunked_prefill=True, max_num_batched_tokens=8192, concurrent_waves=True),
    ),
}


class ScalingWorker(ActivationHookExtension):
    """Adds a forward-only mode: captures are released after each forward, nothing is copied or written."""

    def nh_forward_only(self, enabled):
        if self.nh_waves or self.nh_pending:
            raise RuntimeError("Drain publications before changing the capture mode")
        self.nh_forward_only_enabled = enabled
        return True

    def nh_begin(self, wave_id, request_ids, token_ids, path):
        admitted = super().nh_begin(wave_id, request_ids, token_ids, path)
        if admitted:
            self.nh_waves[wave_id]["forward_only"] = getattr(self, "nh_forward_only_enabled", False)
        return admitted

    def _nh_transfer(self, captured, waves):
        if not getattr(self, "nh_forward_only_enabled", False):
            return super()._nh_transfer(captured, waves)
        captured.clear()  # the next forward reuses this memory
        done = Future()
        done.set_result(0.0)
        return done, None

    @staticmethod
    def _nh_publish(wave):
        if not wave.get("forward_only"):
            return ActivationHookExtension._nh_publish(wave)
        forward = sum(seconds(start, end) * share for start, end, share in wave["forward_events"])
        return dict(
            {key: wave["memory"].get(key, 0) for key in MEMORY_KEYS},
            wave_id=wave["id"],
            path=None,
            bytes=0,
            tokens=wave["tokens"].numel(),
            worker_forward_seconds=forward,
            worker_copy_seconds=0.0,
            write_seconds=0.0,
            validation_seconds=0.0,
            serialization_seconds=0.0,
            actual_forward_batches=wave["forward_batches"],
            topology=wave["topology"],
        )


class ScalingExtractor(VLLMExtractor):
    _worker_extension_cls = "scaling.ScalingWorker"

    def set_forward_only(self, enabled):
        async def update():
            async with self.lock:
                await self.llm.collective_rpc("nh_forward_only", args=(enabled,))

        self._future(update()).result()


def cells(lengths, batches):
    for length in lengths:
        for batch in batches:
            yield length, batch


def wave_inputs(stream, length, batch):
    """Two waves of [batch, length] from the start of the token stream, repeated when the cell needs more.

    Repeated text costs the same as new text: prefix caching is off.
    """
    need = 2 * batch * length
    stream = stream.repeat(-(-need // stream.numel()))
    return list(stream[:need].reshape(2, batch, length))


def remove(receipt):
    path = receipt.get("path")
    if path:
        path = Path(path)
        shutil.rmtree(path) if path.is_dir() else path.unlink(missing_ok=True)


def counters(extractor):
    names = ("backpressure_seconds", "final_drain_seconds", "generate_seconds")
    return {name: getattr(extractor, name, 0.0) for name in names}


async def stream_cell(extractor, waves, directory):
    receipts = []
    async for receipt in extractor.stream_to_disk(waves, directory):
        receipts.append({key: value for key, value in receipt.items() if key not in ("timeline", "scheduler")})
        remove(receipt)
    return receipts


def summarise(receipts, tokens, wall, before, after, saving):
    forward = sum(r["worker_forward_seconds"] for r in receipts)
    copy = sum(r["worker_copy_seconds"] for r in receipts)
    write = sum(r["write_seconds"] for r in receipts)
    written = sum(r.get("bytes", 0) for r in receipts)
    return dict(
        tokens=tokens,
        saving=saving,
        wall_seconds=wall,
        end_to_end_tokens_per_second=tokens / wall,
        forward_seconds=forward,
        forward_tokens_per_second=tokens / forward if forward else None,
        copy_seconds=copy,
        copy_tokens_per_second=tokens / copy if copy else None,
        write_seconds=write,
        write_tokens_per_second=tokens / write if write else None,
        written_bytes=written,
        write_gigabytes_per_second=written / write / 1e9 if write else None,
        backpressure_seconds=after["backpressure_seconds"] - before["backpressure_seconds"],
        final_drain_seconds=after["final_drain_seconds"] - before["final_drain_seconds"],
        forward_batches=[len(r["actual_forward_batches"]) for r in receipts],
        peak_device_memory_bytes=max(r.get("peak_device_memory_bytes", 0) for r in receipts),
    )


def run_vllm(args, options, sites, stream, record):
    max_model_len = max(args.lengths) + 1
    options = dict(options)
    options.setdefault("max_num_batched_tokens", max_model_len)

    def start():
        extractor = ScalingExtractor(
            record["model_path"],
            sites,
            max_model_len=max_model_len,
            sequences_per_call=max(args.batches),
            sink_dir=os.environ.get("NH_SINK_DIR"),
            gpu_memory_utilization=float(os.environ.get("NH_GPU_MEMORY_UTILIZATION", "0.92")),
            kv_cache_memory_bytes=args.kv_cache_bytes,
            publication_layout="per_site",
            precision=args.precision,
            **options,
        )
        # One untimed wave compiles kernels and touches the pinned buffers.
        extractor.set_forward_only(False)
        asyncio.run(stream_cell(extractor, [stream[:1024].reshape(1, 1024)], args.output / "scratch" / "warmup"))
        return extractor

    extractor = start()
    record["engine"] = dict(options, precision_receipt=extractor.precision_receipt)
    try:
        for length, batch in cells(args.lengths, args.batches):
            cell = dict(length=length, batch=batch)
            record["cells"].append(cell)
            if batch * length > args.wave_token_cap:
                cell["status"] = "not run: wave above token cap"
                continue
            saving = batch * length * args.bytes_per_token <= args.saving_cap_bytes
            directory = args.output / "scratch" / f"L{length}-B{batch}"
            try:
                extractor.set_sequences_per_call(batch)
                extractor.set_forward_only(not saving)
                waves = wave_inputs(stream, length, batch)
                before = counters(extractor)
                started = time.perf_counter()
                receipts = asyncio.run(stream_cell(extractor, waves, directory))
                wall = time.perf_counter() - started
                cell.update(summarise(receipts, 2 * batch * length, wall, before, counters(extractor), saving))
                cell["status"] = "ok"
            except Exception as error:
                cell.update(status="memory failure" if is_memory_failure(error) else "error", error=repr(error))
                cell["traceback"] = traceback.format_exc()[-4000:]
                extractor.close()
                extractor = start()  # a failed engine refuses further work
            finally:
                shutil.rmtree(directory, ignore_errors=True)
            write_json(args.output / f"{args.config}-{args.precision}.json", record)
            print(json.dumps({k: cell.get(k) for k in ("length", "batch", "status", "end_to_end_tokens_per_second")}))
    finally:
        extractor.close()


def hf_forward_only(extractor, waves):
    """Forward with hooks capturing on the device; rows are dropped after each forward."""
    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in waves]
    started = time.perf_counter()
    for tokens, (begin, end) in zip(waves, events):
        with extractor._evaluating() as decoder, torch.inference_mode():
            inputs = tokens.to(extractor.device)
            begin.record()
            decoder(input_ids=inputs, use_cache=False)
            end.record()
    torch.cuda.synchronize()
    wall = time.perf_counter() - started
    return wall, sum(seconds(begin, end) for begin, end in events)


def run_hf(args, sites, stream, record):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        record["model_path"],
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
        local_files_only=True,
    )
    with HFExtractor(model, sites, publication_layout="per_site", precision=args.precision) as extractor:
        asyncio.run(stream_cell(extractor, [stream[:1024].reshape(1, 1024)], args.output / "scratch" / "warmup"))
        for length, batch in cells(args.lengths, args.batches):
            cell = dict(length=length, batch=batch)
            record["cells"].append(cell)
            if batch * length > args.wave_token_cap:
                cell["status"] = "not run: wave above token cap"
                continue
            saving = batch * length * args.bytes_per_token <= args.saving_cap_bytes
            directory = args.output / "scratch" / f"L{length}-B{batch}"
            waves = wave_inputs(stream, length, batch)
            torch.cuda.reset_peak_memory_stats()
            try:
                if saving:
                    before = dict(
                        counters(extractor), forward=extractor.model_forward_seconds, copy=extractor.cpu_copy_seconds
                    )
                    started = time.perf_counter()
                    receipts = asyncio.run(stream_cell(extractor, waves, directory))
                    wall = time.perf_counter() - started
                    forward = extractor.model_forward_seconds - before["forward"]
                    copy = extractor.cpu_copy_seconds - before["copy"]
                    for receipt in receipts:  # HF receipts carry no worker timings
                        receipt.update(
                            worker_forward_seconds=forward / len(receipts),
                            worker_copy_seconds=copy / len(receipts),
                            actual_forward_batches=[batch],
                        )
                    cell.update(summarise(receipts, 2 * batch * length, wall, before, counters(extractor), True))
                else:
                    wall, forward = hf_forward_only(extractor, waves)
                    tokens = 2 * batch * length
                    cell.update(
                        tokens=tokens,
                        saving=False,
                        wall_seconds=wall,
                        forward_seconds=forward,
                        end_to_end_tokens_per_second=tokens / wall,
                        forward_tokens_per_second=tokens / forward,
                    )
                cell.update(status="ok", peak_device_memory_bytes=torch.cuda.max_memory_allocated())
            except Exception as error:
                cell.update(status="memory failure" if is_memory_failure(error) else "error", error=repr(error))
                cell["traceback"] = traceback.format_exc()[-4000:]
                if cell["status"] == "error":  # cleanup after a CUDA fault can abort the process
                    print(cell["traceback"], flush=True)
                extractor.captured.clear()
                gc.collect()
                torch.cuda.empty_cache()
            finally:
                shutil.rmtree(directory, ignore_errors=True)
            write_json(args.output / f"{args.config}-{args.precision}.json", record)
            print(json.dumps({k: cell.get(k) for k in ("length", "batch", "status", "end_to_end_tokens_per_second")}))


def run(args):
    torch.set_num_threads(4)
    manifest = json.loads((args.work / "manifest.json").read_text())
    stream = torch.cat([load_tokens(args.work, language, manifest) for language in ("en", "de", "es")])
    sites = [HookSite(**site) for site in json.loads(args.sites.read_text())]
    args.output.mkdir(parents=True, exist_ok=True)
    backend, options = CONFIGS[args.config]
    record = dict(
        config=args.config,
        backend=backend,
        precision=args.precision,
        model=MODEL,
        model_path=manifest["model_path"],
        sites=[site.key for site in sites],
        bytes_per_token=args.bytes_per_token,
        saving_cap_bytes=args.saving_cap_bytes,
        wave_token_cap=args.wave_token_cap,
        kv_cache_bytes=args.kv_cache_bytes,
        gpu=torch.cuda.get_device_name(),
        software=versions(),
        source_hashes=package_hashes(),
        precision_settings=precision_settings(),
        timing="forward: CUDA events around the model forward with hooks; copy: D2H on the side stream; "
        "write: sha256, safetensors, fsync and rename on writer threads. Stages overlap; never add them.",
        cells=[],
    )
    if backend == "vllm":
        run_vllm(args, options, sites, stream, record)
    else:
        run_hf(args, sites, stream, record)
    record["precision_settings_after"] = precision_settings()
    write_json(args.output / f"{args.config}-{args.precision}.json", record)
    shutil.rmtree(args.output / "scratch", ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    runner = sub.add_parser("run")
    runner.add_argument("--config", choices=CONFIGS, required=True)
    runner.add_argument("--precision", choices=("native", "matched"), default="native")
    runner.add_argument("--work", type=Path, required=True)
    runner.add_argument("--sites", type=Path, required=True)
    runner.add_argument("--output", type=Path, required=True)
    runner.add_argument("--lengths", type=int, nargs="+", default=LENGTHS)
    runner.add_argument("--batches", type=int, nargs="+", default=BATCHES)
    runner.add_argument("--bytes-per-token", type=int, required=True, help="Saved bytes per token for all sites")
    runner.add_argument("--saving-cap-bytes", type=int, default=8 * 10**9, help="Largest wave that is saved")
    runner.add_argument("--kv-cache-bytes", type=int, default=8 * 2**30)
    runner.add_argument("--wave-token-cap", type=int, default=WAVE_TOKEN_CAP, help="Skip waves with more tokens")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
