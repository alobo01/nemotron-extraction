"""Replay frozen MoE layers with matched arithmetic and forced expert sets.

The full checkpoint is never inferred here. A two-layer ME fixture retains the
original dimensions, with unused tiny vocabulary tables. The three original
expert layers are loaded serially into its native vLLM expert operator. Frozen
NVIDIA reference classes run in that same worker and share verified weights.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import inspect
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from benchlib.attribution import counterfactual_summary, selection_summary
from benchlib.common import tensor_digest, versions, write_json
from benchlib.precision import checkpoint_source_reference, ieee_precision
from benchlib.stats import Agreement, same_expert_sets
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import nn

from nemotron_extraction.hooks import layers_of

SOURCE_SHA = "52273a23a2c768ffe5938e96f6832afd762029293a8be9bfed886757ba6a39b5"
ANCHOR_SHA = "c5e6f8311596eba310ae9c0f181334dc5d2ae8fcbd8290d7f26ce995b30fb840"
LAYERS = (1, 27, 51)


def metric(a, b):
    value = Agreement()
    value.add(a.detach().cpu(), b.detach().cpu())
    return value.result()


def persist(path, values):
    tensors = {k: v.detach().cpu().contiguous() for k, v in values.items()}
    assert all(bool(torch.isfinite(v).all()) for v in tensors.values())
    save_file(tensors, str(path))
    restored = load_file(str(path))
    assert tensors.keys() == restored.keys()
    assert all(v.dtype == restored[k].dtype and torch.equal(v, restored[k]) for k, v in tensors.items())
    return {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in tensors.items()}


def prepare(root, source, traces):
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(source, trust_remote_code=True, local_files_only=True)
    config.num_hidden_layers, config.hybrid_override_pattern, config.vocab_size = 2, "ME", 128
    torch.manual_seed(7)
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True, torch_dtype=torch.bfloat16)
    index = json.loads((source / "model.safetensors.index.json").read_text())["weight_map"]
    copied = []
    with contextlib.ExitStack() as stack, torch.no_grad():
        files = {}
        for name, target in model.state_dict().items():
            if not name.startswith(("backbone.layers.0.", "backbone.layers.1.")):
                continue
            shard = index[name]
            if shard not in files:
                files[shard] = stack.enter_context(safe_open(str(source / shard), framework="pt"))
            original = files[shard].get_tensor(name)
            target.copy_(original)
            assert torch.equal(target, original.to(target.dtype)), name
            copied.append(name)
    assert "backbone.layers.1.mixer.gate.e_score_correction_bias" in copied
    model.save_pretrained(root / "fixture", safe_serialization=True)
    inputs = {}
    for backend, trace in zip(("hf", "vllm"), traces):
        with safe_open(str(trace), framework="pt") as file:
            tokens = file.get_tensor("token_ids").flatten()
            assert len(tokens) == 2048 and tensor_digest(tokens) == ANCHOR_SHA
            if "token_ids" in inputs:
                assert torch.equal(tokens, inputs["token_ids"])
            inputs["token_ids"] = tokens
            for layer in LAYERS:
                for boundary in ("input", "out"):
                    value = file.get_tensor(f"L{layer:02}_mixer_{boundary}")
                    assert value.shape == (2048, config.hidden_size) and bool(torch.isfinite(value).all())
                    inputs[f"{backend}_L{layer:02}_{boundary}"] = value
    schema = persist(root / "inputs.safetensors", inputs)
    write_json(
        root / "preparation.json",
        {
            "source": str(source),
            "copied": copied,
            "inputs": schema,
            "anchor_sha256": ANCHOR_SHA,
            "software": versions(),
            "scope": __doc__,
        },
    )


class AttributionWorker:
    def attribution_prepare(self, root, source, layer):
        self.attr_root = root = Path(root)
        source = Path(source)
        config = json.loads((source / "config.json").read_text())
        assert config["n_group"] == config["topk_group"] == 1
        assert config["num_experts_per_tok"] == 6 and config["norm_topk_prob"]
        assert config["mlp_hidden_act"] == "relu2" and not config["mlp_bias"]
        _, layers = layers_of(self.model_runner.model)
        self.attr_mixer = mixer = layers[1].mixer
        self.attr_reference = reference = checkpoint_source_reference(root / "reference_model.py", SOURCE_SHA, config)
        params = dict(mixer.named_parameters())
        up = next(p for n, p in params.items() if n.endswith("w13_weight"))
        down = next(p for n, p in params.items() if n.endswith("w2_weight"))
        assert tuple(up.shape) == (config["n_routed_experts"], config["moe_intermediate_size"], config["hidden_size"])
        index = json.loads((source / "model.safetensors.index.json").read_text())["weight_map"]
        prefix = f"backbone.layers.{layer}.mixer."
        verified = []
        with contextlib.ExitStack() as stack, torch.no_grad():
            files = {}

            def read(name):
                key = prefix + name
                shard = index[key]
                if shard not in files:
                    files[shard] = stack.enter_context(safe_open(str(source / shard), framework="pt"))
                return files[shard].get_tensor(key)

            def assign(module, name, target, checkpoint_name, router=False):
                original = read(checkpoint_name)
                target.copy_(original)
                assert torch.equal(target.detach().cpu(), original.to(target.dtype)), checkpoint_name
                value = original.to(device=target.device, dtype=torch.float32) if router else target.detach()
                setattr(module, name, nn.Parameter(value, requires_grad=False))
                verified.append(
                    {
                        "name": checkpoint_name,
                        "checkpoint_dtype": str(original.dtype),
                        "loaded_dtype": str(target.dtype),
                        "sha256": tensor_digest(original),
                    }
                )

            for expert in range(config["n_routed_experts"]):
                assign(reference.experts[expert].up_proj, "weight", up[expert], f"experts.{expert}.up_proj.weight")
                assign(
                    reference.experts[expert].down_proj, "weight", down[expert], f"experts.{expert}.down_proj.weight"
                )
            for projection in ("up_proj", "down_proj"):
                assign(
                    getattr(reference.shared_experts, projection),
                    "weight",
                    getattr(mixer.shared_experts, projection).weight,
                    f"shared_experts.{projection}.weight",
                )
            assign(reference.gate, "weight", mixer.gate.weight, "gate.weight", router=True)
            correction = read("gate.e_score_correction_bias").to(up.device)
            mixer.gate.e_score_correction_bias.copy_(correction)
            reference.gate.e_score_correction_bias = correction.float()
        self.attr_layer = layer
        self.attr_scale = float(config["routed_scaling_factor"])
        reference.eval()
        router_source = inspect.getsource(type(mixer.experts.router))
        (root / "native_router_source.py").write_text(router_source)
        return {
            "layer": layer,
            "verified_parameters": verified,
            "source_sha256": SOURCE_SHA,
            "expert_dtype": str(up.dtype),
            "router_dtype": str(reference.gate.weight.dtype),
            "native_class": type(mixer).__name__,
            "expert_class": type(mixer.experts).__name__,
            "router_class": type(mixer.experts.router).__name__,
            "router_source_sha256": hashlib.sha256(router_source.encode()).hexdigest(),
        }

    def attribution_case(self, length):
        from vllm.forward_context import set_forward_context

        root, layer, ref, mixer = self.attr_root, self.attr_layer, self.attr_reference, self.attr_mixer
        dtype, device = ref.experts[0].up_proj.weight.dtype, ref.gate.weight.device
        with safe_open(str(root / "inputs.safetensors"), framework="pt") as file:
            tokens = file.get_tensor("token_ids")[:length]
            xh, xv = (
                file.get_tensor(f"{backend}_L{layer:02}_input")[:length].to(device=device, dtype=dtype)
                for backend in ("hf", "vllm")
            )
            saved_h, saved_v = (file.get_tensor(f"{backend}_L{layer:02}_out")[:length] for backend in ("hf", "vllm"))
        name = f"{'bf16' if dtype == torch.bfloat16 else 'fp32'}-L{layer:02}-{length}"
        output = root / "cases" / name
        output.mkdir(parents=True, exist_ok=False)
        router = mixer.experts.router
        original_gate, original_selector = mixer.gate.forward, router.select_experts
        observation, forced = {}, [None]

        def aligned_gate(x):
            logits = F.linear(x.float(), ref.gate.weight)
            observation["logits"] = logits.detach().clone()
            return logits, None

        def select(*args, **kwargs):
            weights, ids = original_selector(*args, **kwargs)
            observation.update(ids=ids.detach().clone(), weights=weights.detach().clone())
            if forced[0] is not None:
                if forced[0] == "sham":
                    selected = ids
                elif forced[0] == "aligned":
                    selected = ref.gate.get_topk_indices(observation["logits"].sigmoid())
                else:
                    selected = rh
                scale = weights.sum(-1).mean()
                assert torch.allclose(weights.sum(-1), scale.expand(length), atol=1e-5, rtol=1e-5)
                assert min(abs(float(scale) - 1), abs(float(scale) - self.attr_scale)) < 1e-5
                scores = observation["logits"].sigmoid().gather(1, selected.long())
                weights = (scores / (scores.sum(-1, keepdim=True) + 1e-20) * scale).to(weights.dtype)
                ids = selected.to(ids.dtype)
            observation["returned_ids"] = ids.detach().clone()
            return weights, ids

        def native(x, intervention=None):
            forced[0] = intervention
            with set_forward_context(None, self.model_runner.vllm_config, num_tokens=len(x)):
                value = mixer(x)
            torch.cuda.synchronize()
            return value.detach().clone(), {k: v.detach().clone() for k, v in observation.items()}

        def reference(x, ids):
            scores = F.linear(x.float(), ref.gate.weight).sigmoid().gather(1, ids)
            weights = scores / (scores.sum(-1, keepdim=True) + 1e-20) * self.attr_scale
            return (ref.moe(x, ids, weights) + ref.shared_experts(x)).detach().clone()

        try:
            mixer.gate.forward, router.select_experts = aligned_gate, select
            with ieee_precision() as settings, torch.inference_mode():
                rh, _ = ref.gate(xh)
                rv, _ = ref.gate(xv)
                lh, lv = (F.linear(x.float(), ref.gate.weight) for x in (xh, xv))
                routing, route_rows = selection_summary(
                    rh,
                    rv,
                    lh.sigmoid() + ref.gate.e_score_correction_bias,
                    lv.sigmoid() + ref.gate.e_score_correction_bias,
                )
                h, v = reference(xh, rh), reference(xv, rv)
                forced_v, forced_h = reference(xv, rh), reference(xh, rv)
                nh, oh = native(xh)
                nv, ov = native(xv)
                sham, _ = native(xv, "sham")
                aligned_h, ah = native(xh, "aligned")
                aligned_v, av = native(xv, "aligned")
                nf, _ = native(xv, "reference")
                assert torch.equal(lh, oh["logits"]) and torch.equal(lv, ov["logits"])
                mismatch_h = ~same_expert_sets(rh, oh["ids"])
                mismatch_v = ~same_expert_sets(rv, ov["ids"])
                assert bool(same_expert_sets(rh, ah["returned_ids"]).all())
                assert bool(same_expert_sets(rv, av["returned_ids"]).all())
                corrected_h = lh.sigmoid() + ref.gate.e_score_correction_bias
                corrected_v = lv.sigmoid() + ref.gate.e_score_correction_bias
                native_disagreements = {}
                for label, corrected, mismatch in (("h", corrected_h, mismatch_h), ("v", corrected_v, mismatch_v)):
                    ordered = corrected.sort(-1, descending=True).values
                    gap = ordered[:, 5] - ordered[:, 6]
                    native_disagreements[label] = {
                        "rows": int(mismatch.sum()),
                        "at_exact_cutoff_tie": int((mismatch & (gap == 0)).sum()),
                        "at_gap_at_most_1e_6": int((mismatch & (gap <= 1e-6)).sum()),
                        "maximum_cutoff_gap": float(gap[mismatch].max()) if bool(mismatch.any()) else None,
                    }
                reference_cf, cf_rows = counterfactual_summary(h, v, forced_v, route_rows["changed"])
                native_cf, native_rows = counterfactual_summary(h, aligned_v, nf, route_rows["changed"])
                native_raw_cf, raw_rows = counterfactual_summary(h, sham, nf, ~same_expert_sets(rh, ov["ids"]))
                comparisons = {
                    "same_input_hf": metric(h, nh),
                    "same_input_vllm": metric(v, nv),
                    "sham_vs_raw_native": metric(nv, sham),
                    "same_input_hf_aligned_selector": metric(h, aligned_h),
                    "same_input_vllm_aligned_selector": metric(v, aligned_v),
                    "inherited_input": metric(h, v),
                    "forced_reference_routes": metric(h, forced_v),
                    "reference_vs_saved_hf": metric(saved_h.to(dtype), h.cpu()),
                    "native_vs_saved_vllm": metric(saved_v.to(dtype), nv.cpu()),
                }
                tensors = {
                    "token_ids": tokens,
                    "input_h": xh,
                    "input_v": xv,
                    "reference_h": h,
                    "reference_v": v,
                    "reference_forced_v": forced_v,
                    "reference_forced_h": forced_h,
                    "native_h": nh,
                    "native_v": nv,
                    "native_sham_v": sham,
                    "native_forced_v": nf,
                    "native_aligned_h": aligned_h,
                    "native_aligned_v": aligned_v,
                    "reference_ids_h": rh,
                    "reference_ids_v": rv,
                    "native_ids_h": oh["ids"],
                    "native_ids_v": ov["ids"],
                    "aligned_ids_h": ah["returned_ids"],
                    "aligned_ids_v": av["returned_ids"],
                    "native_mismatch_h": mismatch_h,
                    "native_mismatch_v": mismatch_v,
                    "corrected_scores_h": corrected_h,
                    "corrected_scores_v": corrected_v,
                    **{f"routes_{k}": t for k, t in route_rows.items()},
                    **{f"reference_cf_{k}": t for k, t in cf_rows.items()},
                    **{f"native_cf_{k}": t for k, t in native_rows.items()},
                    **{f"raw_cf_{k}": t for k, t in raw_rows.items()},
                }
                schema = persist(output / "outputs.safetensors", tensors)
                result = {
                    "name": name,
                    "layer": layer,
                    "length": length,
                    "dtype": str(dtype),
                    "precision": settings,
                    "routing": routing,
                    "reference_counterfactual": reference_cf,
                    "native_counterfactual": native_cf,
                    "comparisons": comparisons,
                    "schema": schema,
                    "native_raw_counterfactual": native_raw_cf,
                    "native_router_disagreements": native_disagreements,
                    "exact_native_route_sets": not bool(mismatch_h.any() or mismatch_v.any()),
                    "exact_aligned_route_sets": True,
                    "persisted_exact_finite": True,
                }
                write_json(output / "receipt.json", result)
                print(
                    "ATTRIBUTION_CASE_VERIFIED",
                    name,
                    json.dumps(
                        {
                            "routing": routing,
                            "reference_counterfactual": reference_cf,
                            "native_counterfactual": native_cf,
                            "comparisons": comparisons,
                        }
                    ),
                    flush=True,
                )
                return result
        finally:
            mixer.gate.forward, router.select_experts = original_gate, original_selector


async def run(root, source, profiles=("bfloat16", "float32")):
    from vllm import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    base = dict(
        model=str(root / "fixture"),
        trust_remote_code=True,
        enforce_eager=True,
        tensor_parallel_size=1,
        enable_prefix_caching=False,
        enable_chunked_prefill=True,
        max_model_len=2049,
        max_num_seqs=1,
        max_num_batched_tokens=2049,
        skip_tokenizer_init=True,
        gpu_memory_utilization=0.65,
        kv_cache_memory_bytes=512 * 1024**2,
        mamba_cache_dtype="float32",
        mamba_ssm_cache_dtype="float32",
        worker_extension_cls="moe_attribution.AttributionWorker",
    )
    # Reuse explicitly staged, completed cases without rerunning or overwriting.
    cases = {p.parent.name: json.loads(p.read_text()) for p in (root / "cases").glob("*/receipt.json")}
    reused_cases = sorted(cases)
    weights, controls = {}, {}
    for dtype in profiles:
        controls[dtype] = dict(base, dtype=dtype)
        engine = AsyncLLM.from_engine_args(AsyncEngineArgs(**controls[dtype]))
        try:
            for layer in LAYERS:
                weights[f"{dtype}-L{layer:02}"] = (
                    await engine.collective_rpc("attribution_prepare", args=(str(root), str(source), layer))
                )[0]
                write_json(root / "weight_progress.json", weights)
                for length in (128, 2048) if layer == 1 else (2048,):
                    receipt = (await engine.collective_rpc("attribution_case", args=(length,)))[0]
                    cases[receipt["name"]] = receipt
                    write_json(root / "case_progress.json", cases)
        finally:
            engine.shutdown()
    write_json(
        root / "result.json",
        {
            "status": "complete",
            "cases": cases,
            "weights": weights,
            "config": controls,
            "software": versions(),
            "gpu": torch.cuda.get_device_name(0),
            "source_sha256": SOURCE_SHA,
            "anchor_sha256": ANCHOR_SHA,
            "scope": __doc__,
            "reused_cases": reused_cases,
        },
    )


def verify(root):
    result = json.loads((root / "result.json").read_text())
    expected = {
        f"{dtype}-L{layer:02}-{length}"
        for dtype in ("bf16", "fp32")
        for layer in LAYERS
        for length in ((128, 2048) if layer == 1 else (2048,))
    }
    assert result["status"] == "complete" and set(result["cases"]) == expected
    for name, receipt in result["cases"].items():
        data = load_file(str(root / "cases" / name / "outputs.safetensors"))
        n = receipt["length"]
        assert data["input_h"].shape == data["input_v"].shape == (n, 2688)
        assert data["reference_h"].shape == data["native_v"].shape == (n, 2688)
        assert data["reference_ids_h"].shape == (n, 6)
        assert all(bool(torch.isfinite(x).all()) for x in data.values())
        for side in ("h", "v"):
            same = same_expert_sets(data[f"reference_ids_{side}"], data[f"native_ids_{side}"])
            assert torch.equal(~same, data[f"native_mismatch_{side}"])
            assert int((~same).sum()) == receipt["native_router_disagreements"][side]["rows"]
            assert bool(same_expert_sets(data[f"reference_ids_{side}"], data[f"aligned_ids_{side}"]).all())
        for label, baseline, forced, changed in (
            ("reference", "reference_v", "reference_forced_v", data["routes_changed"]),
            ("native", "native_aligned_v", "native_forced_v", data["routes_changed"]),
            (
                "native_raw",
                "native_sham_v",
                "native_forced_v",
                ~same_expert_sets(data["reference_ids_h"], data["native_ids_v"]),
            ),
        ):
            cf, _ = counterfactual_summary(data["reference_h"], data[baseline], data[forced], changed)
            # GPU and CPU reduction trees need not produce identical FP64 scalars.
            # Counts and structure stay exact; tolerate only final-bit sum changes.
            saved_cf = receipt[label + "_counterfactual"]
            for key, value in cf.items():
                if isinstance(value, float):
                    assert math.isclose(value, saved_cf[key], rel_tol=1e-12, abs_tol=1e-12), (label, key)
                else:
                    assert value == saved_cf[key], (label, key)
        assert receipt["routing"]["changed_rows"] == int(data["routes_changed"].sum())
        assert receipt["routing"]["changed_rows_satisfying_stability_bound"] == receipt["routing"]["changed_rows"]
        assert receipt["persisted_exact_finite"] and receipt["exact_aligned_route_sets"]
        print("VERIFIED", name, json.dumps({"routing": receipt["routing"], "counterfactual": cf}), flush=True)
    print("MOE_ATTRIBUTION_ARTIFACTS_VERIFIED", len(expected), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "verify"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--traces", nargs=2, type=Path)
    parser.add_argument("--profiles", nargs="+", choices=("bfloat16", "float32"), default=("bfloat16", "float32"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.action == "prepare":
        assert hashlib.sha256((args.root / "reference_model.py").read_bytes()).hexdigest() == SOURCE_SHA
        prepare(args.root, args.source, args.traces)
    elif args.action == "run":
        asyncio.run(run(args.root, args.source, args.profiles))
    else:
        verify(args.root)


if __name__ == "__main__":
    main()
