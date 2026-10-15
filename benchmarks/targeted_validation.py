# /// script
# requires-python = ">=3.10"
# dependencies = ["nemotron-extraction[hf,vllm,benchmark]==1.0.0"]
# ///
"""Small same-forward control, transport validation and GPU utilisation recording.

Run each backend sequentially, then report. Reference inputs are unchanged
snapshots of the sibling's worker.py, hf.py and attention.py capture sources.
The installed hook package has no dependency on those sources.
"""
from __future__ import annotations

import argparse
import ast
import asyncio
from contextlib import contextmanager
from functools import partial
import json
from pathlib import Path
import subprocess
import threading
import time
from types import SimpleNamespace

import torch
from torch import nn
from safetensors.torch import load_file

import nemotron_extraction
from nemotron_extraction import HFExtractor, VLLMExtractor, HookSite
from nemotron_extraction.hooks import representative_sites, resolve
from nemotron_extraction.worker import ActivationHookExtension
from validate import Agreement, sha256, versions, write_json


def reference_namespace(source, *, worker=False, attention=None):
    """Execute only capture definitions; never import training or transport code."""
    scope = {"torch": torch, "nn": nn, "contextmanager": contextmanager}
    if attention is not None:
        scope.update(reference_namespace(attention))
    tree = ast.parse(source)
    nodes = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    if worker:
        nodes.extend(node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name in ("_arg", "_residual"))
        original = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                        and node.name == "ActivationHookExtension")
        original.body = [node for node in original.body if isinstance(node, ast.FunctionDef)
                         and node.name in ("_sae_capture_resid_pre", "_sae_capture_resid_post",
                                           "_sae_capture_output", "_sae_capture_input")]
        if len(original.body) != 4:
            raise ValueError("Reference worker must define all four original capture callbacks")
        nodes.append(original)
    else:
        nodes.extend(node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)))
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 "<reference-capture-source>", "exec"), scope)
    return scope


def exact(reference, candidate):
    if not reference or reference.keys() != candidate.keys():
        raise ValueError("Control hook sites differ")
    result = {}
    for key, left in reference.items():
        right = candidate[key]
        if left.shape != right.shape or left.dtype != right.dtype:
            raise ValueError(f"{key}: control shape/dtype mismatch")
        if not bool(torch.isfinite(left).all() and torch.isfinite(right).all()):
            raise ValueError(f"{key}: nonfinite control")
        result[key] = bool(torch.equal(left, right))
    return result


class AuditWorker(ActivationHookExtension):
    """Diagnostic-only extension: independent callbacks on the same GPU forward."""
    def ta_enable(self, source):
        if getattr(self, "ta_handles", None):
            raise RuntimeError("Control already enabled")
        namespace = reference_namespace(source, worker=True)
        self.ta_reference = namespace["ActivationHookExtension"]()
        self.ta_reference.sae_captured = {}
        self.ta_handles = []
        for key, info in self.nh_topology.items():
            site = HookSite(info["layer"], info["kind"])
            _, module = resolve(self.model_runner.model, site)
            if site.kind in ("resid_pre", "attn_out_prelinear"):
                callback = (self.ta_reference._sae_capture_resid_pre if site.kind == "resid_pre"
                            else self.ta_reference._sae_capture_input)
                handle = module.register_forward_pre_hook(partial(callback, key), with_kwargs=True)
            else:
                callback = (self.ta_reference._sae_capture_resid_post if site.kind == "resid_post"
                            else self.ta_reference._sae_capture_output)
                handle = module.register_forward_hook(partial(callback, key))
            self.ta_handles.append(handle)
        self.ta_enabled = True
        return True

    def ta_disable(self):
        for handle in getattr(self, "ta_handles", []):
            handle.remove()
        self.ta_handles = []
        self.ta_enabled = False
        return True

    def _nh_end(self, module, args, output):
        if getattr(self, "ta_enabled", False):
            wave = self.nh_active
            comparisons = exact(self.ta_reference.sae_captured, self.nh_captured)
            wave.setdefault("callback_checks", []).append(comparisons)
            # Independent blocking copy of these exact captures, before buffer
            # reuse. This deliberately slow control is excluded from profiling.
            shadow = wave.setdefault("shadow", {})
            length = wave["tokens"].shape[1]
            for key, source in self.ta_reference.sae_captured.items():
                host = shadow.setdefault(key, torch.empty((wave["tokens"].numel(), source.shape[-1]),
                                                         dtype=source.dtype))
                for index, start, stop in self.nh_spans:
                    host[index * length:(index + 1) * length].copy_(source[start:stop].cpu())
            self.ta_reference.sae_captured.clear()
        super()._nh_end(module, args, output)

    @staticmethod
    def _nh_publish(wave):
        receipt = ActivationHookExtension._nh_publish(wave)
        if "shadow" in wave:
            receipt["same_forward_callbacks"] = wave["callback_checks"]
            receipt["blocking_vs_async"] = exact(wave["shadow"], {
                key: wave["tensors"][key] for key in wave["shadow"]})
        return receipt

    def nh_remove(self):
        self.ta_disable()
        return super().nh_remove()


class AuditExtractor(VLLMExtractor):
    _worker_extension_cls = "targeted_validation.AuditWorker"


class Utilisation:
    """200 ms GPU samples, phase-labelled; utilisation is sampled, not kernel time."""
    def __init__(self, output):
        self.output, self.phase, self.samples = output, "startup", []
        self.started = time.perf_counter()
        self.errors = []

    def __enter__(self):
        device = str(torch.cuda.get_device_properties(0).uuid)
        if not device.startswith("GPU-"):
            device = "GPU-" + device
        self.process = subprocess.Popen([
            "nvidia-smi", "-i", device,
            "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,power.draw",
            "--format=csv,noheader,nounits", "--loop-ms=200"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()
        return self

    def _read(self):
        with self.output.open("w") as stream:
            for line in self.process.stdout:
                try:
                    values = [float(part.strip()) for part in line.split(",")]
                    if len(values) != 5:
                        raise ValueError("Unexpected nvidia-smi column count")
                    sample = dict(zip(("gpu_percent", "memory_busy_percent", "memory_used_mib",
                                       "memory_total_mib", "power_watts"), values))
                    sample.update(seconds=time.perf_counter() - self.started, phase=self.phase)
                    self.samples.append(sample)
                    stream.write(json.dumps(sample) + "\n")
                    stream.flush()
                except ValueError as error:
                    self.errors.append(str(error))

    def __exit__(self, *exc):
        self.process.terminate()
        self.thread.join(timeout=10)
        self.process.wait(timeout=10)
        self.errors.extend(line for line in self.process.stderr.read().splitlines() if line)

    def summary(self):
        result = {}
        for phase in sorted({s["phase"] for s in self.samples}):
            samples = [s for s in self.samples if s["phase"] == phase]
            result[phase] = {"samples": len(samples), "sample_interval_seconds": .2}
            for key in ("gpu_percent", "memory_busy_percent", "memory_used_mib", "power_watts"):
                values = torch.tensor([s[key] for s in samples], dtype=torch.float64)
                result[phase][key] = {"mean": float(values.mean()), "p95": float(values.quantile(.95)),
                                      "maximum": float(values.max())}
        return {"phases": result, "errors": self.errors}


def overlap_seconds(timeline, name):
    """Union of transfer/write intervals intersecting any forward interval."""
    spans = sorted((max(a["start"], b["start"]), min(a["end"], b["end"]))
                   for a in timeline if a["name"] == name
                   for b in timeline if b["name"] == "forward"
                   if max(a["start"], b["start"]) < min(a["end"], b["end"]))
    total, end = 0., float("-inf")
    for start, stop in spans:
        total += max(0., stop - max(start, end))
        end = max(end, stop)
    return total


def control_batches(tokens):
    rows = tokens[:4 * 2048].reshape(4, 2048)[[2, 0, 2, 1]].clone()
    return [rows[:, :128].contiguous(), rows, rows[:1, :37].contiguous()]


async def persist(extractor, batches, output):
    return [receipt async for receipt in extractor.stream_to_disk(batches, output)]


def run(args):
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.work / "manifest.json").read_text())
    token_path = args.work / "en.safetensors"
    if sha256(token_path) != manifest["languages"]["en"]["token_sha256"]:
        raise ValueError("Frozen input checksum mismatch")
    tokens = load_file(str(token_path))["token_ids"]
    controls = control_batches(tokens)
    reference = {name: (args.reference / name).read_text() for name in ("hf.py", "worker.py", "attention.py")}
    source_hashes = {name: sha256(args.reference / name) for name in reference}
    result = {"backend": args.backend, "model_revision": manifest["model_revision"],
              "tokenizer_revision": manifest["tokenizer_revision"],
              "token_sha256": sha256(token_path), "reference_source_sha256": source_hashes,
              "script_sha256": sha256(__file__), "software": versions(), "controls": [],
              "package_code_sha256": {p.name: sha256(p) for p in sorted(
                  Path(nemotron_extraction.__file__).parent.glob("*.py"))},
              "memory_target_fraction": .96, "status": "running"}
    extractor = None
    monitor = Utilisation(args.output / f"{args.backend}-utilisation.jsonl")
    try:
        with monitor:
            started = time.perf_counter()
            sites_file = args.output / "sites.json"
            if args.backend == "hf":
                from transformers import AutoModelForCausalLM
                model = AutoModelForCausalLM.from_pretrained(manifest["model_path"], torch_dtype=torch.bfloat16,
                    device_map="cuda", trust_remote_code=True, local_files_only=True)
                sites = representative_sites(model)
                extractor = HFExtractor(model, sites)
                write_json(sites_file, [{"layer": s.layer, "kind": s.kind} for s in sites])
            else:
                sites = [HookSite(**s) for s in json.loads(sites_file.read_text())]
                extractor = AuditExtractor(manifest["model_path"], sites, sequences_per_call=4,
                    max_model_len=2049, sink_dir=str(args.output), gpu_memory_utilization=.96)
            result["startup_seconds"] = time.perf_counter() - started
            result["hook_install_seconds"] = extractor.hook_install_seconds
            monitor.phase = "warmup"
            extractor.extract(controls[1])
            monitor.phase = "controls"
            if args.backend == "vllm":
                extractor._future(extractor.llm.collective_rpc("ta_enable", args=(reference["worker.py"],))).result()
            for index, batch in enumerate(controls):
                path = args.output / "controls" / args.backend / str(index)
                if args.backend == "hf":
                    scope = reference_namespace(reference["hf.py"], attention=reference["attention.py"])
                    requests = {s.key: (SimpleNamespace(kind=s.kind, layer=s.layer, head_idx=None),
                                       SimpleNamespace(module_path=extractor.topology[s.key]["module_path"], extra={}))
                                for s in sites}
                    with scope["capture_many"](extractor.model, requests, device="cpu") as buffers:
                        receipts = asyncio.run(persist(extractor, [batch], path))
                    saved = load_file(receipts[0]["path"])
                    comparison = exact({k: b.consume() for k, b in buffers.items()}, {s.key: saved[s.key] for s in sites})
                    receipts[0].update(same_forward_callbacks=[comparison], blocking_vs_async=comparison)
                else:
                    receipts = asyncio.run(persist(extractor, [batch], path))
                if not torch.equal(load_file(receipts[0]["path"])["token_ids"], batch):
                    raise ValueError("Control token order changed")
                result["controls"].append(receipts[0])
                write_json(args.output / f"{args.backend}.json", result)
            if args.backend == "vllm":
                extractor._future(extractor.llm.collective_rpc("ta_disable")).result()
            # Original callbacks and blocking shadows are absent in this phase.
            monitor.phase = "pipeline"
            batches = [tokens[i * 8192:(i + 1) * 8192].reshape(4, 2048) for i in range(args.batches)]
            started = time.perf_counter()
            receipts = asyncio.run(persist(extractor, batches, args.output / "pipeline" / args.backend))
            seconds = time.perf_counter() - started
            timeline = [event for receipt in receipts for event in receipt["timeline"]]
            forward = sum(e["end"] - e["start"] for e in timeline if e["name"] == "forward")
            result["pipeline"] = {"tokens": sum(b.numel() for b in batches), "batches": len(batches),
                "batch_size": 4, "wall_seconds": seconds, "forward_seconds": forward,
                "completed_to_disk_tokens_per_second": sum(b.numel() for b in batches) / seconds,
                "forward_tokens_per_second": sum(b.numel() for b in batches) / forward,
                "D2H_forward_overlap_seconds": overlap_seconds(timeline, "D2H"),
                "write_forward_overlap_seconds": overlap_seconds(timeline, "write"), "receipts": receipts}
            result["topology"] = extractor.topology
            result["actual_forward_batches"] = extractor.actual_forward_batches
            result["peak_device_memory_bytes"] = extractor.peak_device_memory_bytes
            result["backpressure_seconds"] = extractor.backpressure_seconds
            result["final_drain_seconds"] = extractor.final_drain_seconds
            monitor.phase = "cleanup"
            extractor.close()
            result["status"] = "complete"
        if monitor.errors or not any(s["phase"] == "pipeline" for s in monitor.samples):
            raise RuntimeError(f"GPU utilisation recording failed: {monitor.errors}")
    except BaseException as error:
        result.update(status="error", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        if extractor is not None:
            extractor.close()
        result["utilisation"] = monitor.summary()
        write_json(args.output / f"{args.backend}.json", result)
    print(json.dumps({"backend": args.backend, "status": result["status"], "pipeline": {
        k: v for k, v in result["pipeline"].items() if k != "receipts"}}), flush=True)


def report(args):
    records = {b: json.loads((args.output / f"{b}.json").read_text()) for b in ("hf", "vllm")}
    if any(r["status"] != "complete" for r in records.values()):
        raise ValueError("Both targeted backends must complete")
    for key in ("model_revision", "tokenizer_revision", "token_sha256", "reference_source_sha256",
                "script_sha256", "package_code_sha256"):
        if records["hf"][key] != records["vllm"][key]:
            raise ValueError(f"Provenance mismatch: {key}")
    callbacks, transport = {}, {}
    agreement = {}
    for backend, record in records.items():
        callbacks[backend] = all(all(check.values()) for r in record["controls"] for check in r["same_forward_callbacks"])
        transport[backend] = all(all(r["blocking_vs_async"].values()) for r in record["controls"])
    for index in range(3):
        left = load_file(str(args.output / "controls/hf" / str(index) / "000000.safetensors"))
        right = load_file(str(args.output / "controls/vllm" / str(index) / "000000.safetensors"))
        if not torch.equal(left.pop("token_ids"), right.pop("token_ids")):
            raise ValueError("Cross-backend token IDs differ")
        for key in left:
            metric = Agreement()
            metric.add(left[key], right[key])
            agreement[f"control{index}/{key}"] = metric.result()
    result = {"same_forward_callbacks_passed": callbacks, "blocking_vs_async_passed": transport,
              "cross_backend_passed": all(v["passed"] for v in agreement.values()),
              "agreement": agreement, "backends": records}
    write_json(args.output / "results.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("agreement", "backends")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("backend", "report"))
    parser.add_argument("--backend", choices=("hf", "vllm"))
    parser.add_argument("--work", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batches", type=int, default=8)
    args = parser.parse_args()
    if args.action == "backend" and (not args.backend or not args.work or not args.reference or not 2 <= args.batches <= 32):
        parser.error("backend requires --backend, --work, --reference and 2 <= --batches <= 32")
    (run if args.action == "backend" else report)(args)
