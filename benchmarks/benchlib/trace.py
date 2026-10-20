"""Anchor-row traces at every decoder boundary, captured before in-place mutation."""

from __future__ import annotations

import inspect
from functools import partial
from pathlib import Path

import torch
from torch import nn

from nemotron_extraction.hooks import argument, layers_of, mixer_kind, projection_of, tensor_output

from .common import digest, tensor_digest


def runtime_config(model, runner=None):
    """Runtime values and source hashes, rather than inferred library defaults."""
    sources, modules = {}, {}
    for name, module in model.named_modules():
        path = inspect.getsourcefile(type(module))
        if path and path not in sources and Path(path).is_file():
            sources[path] = digest(path)
        interesting = {
            key: str(getattr(module, key))
            for key in (
                "ssm_state_dtype",
                "conv_state_dtype",
                "mamba_ssm_cache_dtype",
                "mamba_cache_dtype",
                "attn_backend",
                "quant_method",
                "scoring_func",
                "top_k",
                "routed_scaling_factor",
            )
            if hasattr(module, key)
        }
        if interesting:
            modules[name] = dict(type=type(module).__name__, **interesting)
    config = getattr(model, "vllm_config", getattr(runner, "vllm_config", None))
    state_dtypes = None
    if config is not None and hasattr(model, "get_mamba_state_dtype_from_config"):
        state_dtypes = [str(value) for value in model.get_mamba_state_dtype_from_config(config)]
    return dict(
        model_class=type(model).__name__,
        sources=sources,
        modules=modules,
        weight_dtypes=sorted({str(p.dtype) for p in model.parameters()}),
        parameter_bytes=sum(p.numel() * p.element_size() for p in model.parameters()),
        device_memory=(
            dict(zip(("free_bytes", "total_bytes"), torch.cuda.mem_get_info())) if torch.cuda.is_available() else None
        ),
        cache_config=str(getattr(config, "cache_config", None)),
        resolved_mamba_state_dtypes=state_dtypes,
        model_config=str(getattr(config, "model_config", None)),
        # vLLM's worker RPC serializer only accepts builtin strings here;
        # TorchVersion is a str subclass that it rejects by exact type.
        torch_version=str(torch.__version__),
        cuda_version=torch.version.cuda,
        tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
        tf32_cudnn=torch.backends.cudnn.allow_tf32,
        matmul_precision=torch.get_float32_matmul_precision(),
    )


class TraceHooks:
    """Capture anchor rows at explicit mathematical boundaries, before mutation."""

    def __init__(self, model, worker=False):
        self.model, self.worker = model, worker
        self.handles, self.values, self.topology = [], {}, {}
        self.span = None
        self.replay = None
        self.detail_layer = None
        self.detail_handles = []
        self.detail_observability = {}
        self.restore_methods = []
        base, layers = layers_of(model)
        for name, module in model.named_modules():
            if name in (
                "model.embed_tokens",
                "backbone.embeddings",
                "backbone.embed_tokens",
                "embed_tokens",
                "embeddings",
            ):
                self.handles.append(module.register_forward_hook(partial(self.post, "embedding", name)))
                break
        else:
            raise ValueError("Cannot resolve embedding module")
        for i, block in enumerate(layers):
            path = f"{base}.{i}"
            self.handles.append(block.register_forward_pre_hook(partial(self.block_pre, i, path), with_kwargs=True))
            self.handles.append(block.register_forward_hook(partial(self.block_post, i, path)))
            self.handles.append(
                block.mixer.register_forward_pre_hook(partial(self.mixer_pre, i, path + ".mixer"), with_kwargs=True)
            )
            self.handles.append(
                block.mixer.register_forward_hook(partial(self.post, f"L{i:02d}_mixer_out", path + ".mixer"))
            )
            self.handles.append(
                block.norm.register_forward_pre_hook(partial(self.norm_pre, i, path + ".norm"), with_kwargs=True)
            )
            if mixer_kind(block.mixer) == "attn_out":
                name, projection = projection_of(block.mixer)
                self.handles.append(
                    projection.register_forward_pre_hook(
                        partial(self.input_hook, f"L{i:02d}_attn_prelinear", path + ".mixer." + name), with_kwargs=True
                    )
                )

    def rows(self, value):
        if self.span is None:
            return None
        start, stop = self.span
        return value.reshape(-1, value.shape[-1])[start:stop]

    def keep(self, key, path, value):
        value = self.rows(value)
        if value is None:
            return
        self.keep_rows(key, path, value)

    def keep_rows(self, key, path, value):
        if key in self.values:
            raise RuntimeError("Anchor captured twice at " + key)
        self.values[key] = value.detach().clone()
        self.topology[key] = dict(module_path=path, width=value.shape[-1], dtype=str(value.dtype))

    def block_pre(self, layer, path, module, args, kwargs):
        x = argument(args, kwargs, 1 if self.worker else 0, "hidden_states")
        if self.worker:
            r = argument(args, kwargs, 2, "residual")
            self.keep(f"L{layer:02d}_raw_hidden", path, x)
            if r is not None:
                self.keep(f"L{layer:02d}_raw_residual", path, r)
            if self.span is not None:
                self.keep_rows(
                    f"L{layer:02d}_resid_pre", path, self.rows(x) if r is None else self.rows(x) + self.rows(r)
                )
        else:
            self.keep(f"L{layer:02d}_resid_pre", path, x)

    def block_post(self, layer, path, module, args, output):
        if self.worker:
            self.keep(f"L{layer:02d}_raw_out_hidden", path, output[0])
            self.keep(f"L{layer:02d}_raw_out_residual", path, output[1])
            if self.span is not None:
                self.keep_rows(f"L{layer:02d}_resid_post", path, self.rows(output[0]) + self.rows(output[1]))
            return
        else:
            value = tensor_output(output)
        self.keep(f"L{layer:02d}_resid_post", path, value)

    def mixer_pre(self, layer, path, module, args, kwargs):
        # Attention mixers take positions before hidden_states in vLLM.
        index = 1 if self.worker and mixer_kind(module) == "attn_out" else 0
        value = argument(args, kwargs, index, "hidden_states")
        self.keep(f"L{layer:02d}_mixer_input", path, value)
        if (
            self.replay is not None
            and self.replay.get("operation", "mixer") == "mixer"
            and layer == self.replay["layer"]
            and self.span is not None
        ):
            value = value.clone()
            replacement = self.replay["input"].to(device=value.device, dtype=value.dtype)
            start, stop = self.span
            value.reshape(-1, value.shape[-1])[start:stop].copy_(replacement)
            if "hidden_states" in kwargs:
                kwargs = dict(kwargs, hidden_states=value)
            else:
                args = list(args)
                args[index] = value
                args = tuple(args)
            self.keep(f"L{layer:02d}_replay_input", path, value)
            return args, kwargs

    def norm_pre(self, layer, path, module, args, kwargs):
        if (
            self.replay is None
            or self.replay.get("operation") != "norm"
            or layer != self.replay["layer"]
            or self.span is None
        ):
            return
        x = argument(args, kwargs, 0, "x" if self.worker else "hidden_states").clone()
        start, stop = self.span
        x.reshape(-1, x.shape[-1])[start:stop].copy_(self.replay["input"].to(x))
        args = list(args)
        kwargs = dict(kwargs)
        name = "x" if self.worker else "hidden_states"
        if name in kwargs:
            kwargs[name] = x
        else:
            args[0] = x
        if self.worker:
            r = kwargs.get("residual", args[1] if len(args) > 1 else None)
            if r is not None:
                r = r.clone()
                r.reshape(-1, r.shape[-1])[start:stop].zero_()
                if "residual" in kwargs:
                    kwargs["residual"] = r
                else:
                    args[1] = r
        self.keep(f"L{layer:02d}_replay_input", path, x)
        return tuple(args), kwargs

    def input_hook(self, key, path, module, args, kwargs):
        self.keep(key, path, argument(args, kwargs, 0, "hidden_states"))

    def details(self, layer):
        """Observe executed submodules; never claim fused internals were observed."""
        self.detail_layer = layer
        base, layers = layers_of(self.model)
        mixer = layers[layer].mixer
        wanted = (
            "in_proj",
            "out_proj",
            "o_proj",
            "q_proj",
            "k_proj",
            "v_proj",
            "qkv_proj",
            "conv1d",
            "norm",
            "gate",
            "shared_experts",
            "experts",
        )
        self.detail_observability = dict(
            mixer_type=type(mixer).__name__,
            requested=list(wanted),
            observed=[],
            routing_ids="unobserved",
            state_policy="new request, position zero; no reused request cache",
        )
        self.detail_observability["parameters"] = {
            name: dict(shape=list(p.shape), dtype=str(p.dtype), sha256=tensor_digest(p))
            for name, p in mixer.named_parameters()
        }

        def capture(name, module, args, output):
            values = output if isinstance(output, tuple) else (output,)
            for j, value in enumerate(values):
                if not isinstance(value, torch.Tensor) or value.ndim not in (2, 3) or self.span is None:
                    continue
                # Only [tokens,width] or [batch,sequence,width], never [batch,width,sequence].
                row_count = value.numel() // value.shape[-1]
                if row_count < self.span[1]:
                    continue
                key = f"L{layer:02d}_detail_{name.replace('.', '-')}_{j}"
                self.keep(key, f"{base}.{layer}.mixer.{name}", value)
                self.detail_observability["observed"].append(name)

        for name, module in mixer.named_modules():
            if name and name.split(".")[-1] in wanted:
                # Convolution rows are channel-major; record why they are not captured.
                if isinstance(module, nn.Conv1d):
                    self.detail_observability["convolution"] = (
                        "functional/channel-major convolution requires kernel-level replay"
                    )
                    continue
                self.detail_handles.append(module.register_forward_hook(partial(capture, name)))
        router = getattr(getattr(mixer, "experts", None), "router", None)
        select = getattr(router, "select_experts", None)
        if callable(select):

            def wrapped(*args, **kwargs):
                result = select(*args, **kwargs)
                capture("actual_router_selection", router, args, result)
                self.detail_observability["routing_ids"] = (
                    "captured select_experts return; tensor dtypes identify IDs and weights"
                )
                return result

            router.select_experts = wrapped
            self.restore_methods.append((router, "select_experts", select))

    def post(self, key, path, module, args, output):
        self.keep(key, path, tensor_output(output))

    def reset(self):
        self.values.clear()
        self.span = None

    def close(self):
        for owner, name, method in self.restore_methods:
            setattr(owner, name, method)
        for handle in self.detail_handles:
            handle.remove()
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.reset()
