"""Shared Nemotron topology and capture semantics, adapted from nemotron-sae."""
from __future__ import annotations
from functools import partial
import torch
from torch import nn
from .types import HookSite, KINDS

def layers_of(model: nn.Module):
    for path in ("model.layers", "backbone.layers", "layers"):
        try:
            layers = model.get_submodule(path)
            if len(layers):
                return path, layers
        except (AttributeError, TypeError):
            continue
    raise ValueError("Unsupported topology: expected Nemotron decoder layers")

def mixer_kind(mixer: nn.Module) -> str | None:
    name = type(mixer).__name__.lower()
    if "mamba" in name or "ssm" in name:
        return "mamba_out"
    if "attention" in name or "attn" in name:
        return "attn_out"
    if "moe" in name or hasattr(mixer, "experts"):
        return "moe_out"
    return None

def projection_of(mixer: nn.Module) -> tuple[str, nn.Module]:
    for name in ("o_proj", "out_proj", "dense"):
        projection = getattr(mixer, name, None)
        if isinstance(projection, nn.Module):
            return name, projection
    raise ValueError(f"Cannot locate attention output projection in {type(mixer).__name__}")

def resolve(model: nn.Module, site: HookSite) -> tuple[str, nn.Module]:
    path, layers = layers_of(model)
    if site.layer >= len(layers):
        raise ValueError(f"{site.key}: model only has {len(layers)} layers")
    block = layers[site.layer]
    path = f"{path}.{site.layer}"
    if site.kind.startswith("resid_"):
        return path, block
    mixer = getattr(block, "mixer", None)
    expected = "attn_out" if site.kind == "attn_out_prelinear" else site.kind
    if mixer is None or mixer_kind(mixer) != expected:
        raise ValueError(f"{site.key}: layer does not have a compatible {expected} mixer")
    if site.kind == "attn_out_prelinear":
        name, projection = projection_of(mixer)
        return f"{path}.mixer.{name}", projection
    return f"{path}.mixer", mixer

def representative_sites(model: nn.Module) -> list[HookSite]:
    _, layers = layers_of(model)
    result = []
    for kind in KINDS:
        eligible = []
        for layer in range(len(layers)):
            try:
                resolve(model, HookSite(layer, kind))
                eligible.append(layer)
            except ValueError:
                pass
        if not eligible:
            raise ValueError(f"No eligible layers for {kind}")
        for layer in sorted({eligible[0], eligible[(len(eligible) - 1) // 2], eligible[-1]}):
            result.append(HookSite(layer, kind))
    return result

def argument(args, kwargs, index, name):
    return kwargs[name] if name in kwargs else args[index]

def tensor_output(output):
    return output[0] if isinstance(output, tuple) else output

def residual(hidden: torch.Tensor, running: torch.Tensor | None = None) -> torch.Tensor:
    return hidden.detach().clone() if running is None else (hidden + running).detach()

def install(model, sites, captured, *, worker=False, capture_events=None):
    """Capture detached GPU copies; callers move them only after the forward pass."""
    handles, topology = [], {}

    def start(value):
        if capture_events is None or not value.is_cuda:
            return None
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    def store(site, value, running=None):
        event = start(value)
        captured[site.key] = residual(value, running)
        if event is not None:
            ended = torch.cuda.Event(enable_timing=True)
            ended.record()
            capture_events.append((site.key, event, ended))

    def pre(site, module, args, kwargs):
        # vLLM decoder layers receive (positions, hidden_states, residual).
        if worker and site.kind == "resid_pre":
            store(site, argument(args, kwargs, 1, "hidden_states"), argument(args, kwargs, 2, "residual"))
        else:
            store(site, argument(args, kwargs, 0, "hidden_states"))

    def post(site, module, args, output):
        if worker and site.kind == "resid_post":
            store(site, *output)
        else:
            store(site, tensor_output(output))

    try:
        for site in sites:
            path, module = resolve(model, site)
            topology[site.key] = {"layer": site.layer, "kind": site.kind, "module_path": path}
            handle = (module.register_forward_pre_hook(partial(pre, site), with_kwargs=True)
                      if site.kind in ("resid_pre", "attn_out_prelinear")
                      else module.register_forward_hook(partial(post, site)))
            handles.append(handle)
    except Exception:
        for handle in handles:
            handle.remove()
        raise
    return handles, topology
