"""Matched arithmetic for comparing the PyTorch reference with vLLM.

"native" keeps each backend's defaults. "matched" removes the differences that
are configuration rather than kernels: IEEE float32 matmuls (no TF32 or reduced
bf16 reductions) on both sides, and on vLLM a float32 MoE router computed from
the checkpoint's float32 gate weights, as the reference model does.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

PRECISIONS = ("native", "matched")


def validate_precision(precision):
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}")
    return precision


def fp32_backends():
    """Backends exposing ``fp32_precision``; resolved lazily because older torch lacks them."""
    return (
        ("global", torch.backends),
        ("matmul", torch.backends.cuda.matmul),
        ("cudnn", torch.backends.cudnn),
        ("conv", torch.backends.cudnn.conv),
        ("rnn", torch.backends.cudnn.rnn),
    )


def precision_settings():
    # Do not mix the legacy TF32 API with the API introduced after torch 2.9.
    modern = tuple(int(x) for x in str(torch.__version__).split("+")[0].split(".")[:2]) >= (2, 9)
    if modern:
        settings = {name: owner.fp32_precision for name, owner in fp32_backends()}
    else:
        settings = dict(matmul_tf32=torch.backends.cuda.matmul.allow_tf32, cudnn_tf32=torch.backends.cudnn.allow_tf32)
    settings.update(
        api="modern" if modern else "legacy",
        bf16_reduced_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
    )
    return settings


def set_precision(settings):
    if settings["api"] == "modern":
        for name, owner in fp32_backends():
            owner.fp32_precision = settings[name]
    else:
        torch.backends.cuda.matmul.allow_tf32 = settings["matmul_tf32"]
        torch.backends.cudnn.allow_tf32 = settings["cudnn_tf32"]
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = settings["bf16_reduced_reduction"]


def ieee_settings():
    settings = precision_settings()
    if settings["api"] == "modern":
        settings.update({name: "ieee" for name, _ in fp32_backends()})
    else:
        settings.update(matmul_tf32=False, cudnn_tf32=False)
    settings["bf16_reduced_reduction"] = False
    return settings


def checkpoint_directory(model, revision=None):
    if Path(model).is_dir():
        return Path(model)
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model, revision=revision, local_files_only=True))


def fp32_router(layers, checkpoint_dir, prefix="backbone.layers"):
    """Compute vLLM MoE router logits in float32 from the checkpoint's gate weights.

    vLLM stores the gate in the model dtype, so the float32 checkpoint values are
    reloaded rather than upcast. Returns the patched layer indices and the
    original forwards for restoring.
    """
    from safetensors import safe_open

    checkpoint_dir = Path(checkpoint_dir)
    index = json.loads((checkpoint_dir / "model.safetensors.index.json").read_text())["weight_map"]
    patched = {}
    for number, layer in enumerate(layers):
        gate = getattr(getattr(layer, "mixer", None), "gate", None)
        if gate is None or not hasattr(gate, "weight"):
            continue
        name = f"{prefix}.{number}.mixer.gate.weight"
        with safe_open(str(checkpoint_dir / index[name]), framework="pt") as file:
            weight = file.get_tensor(name).to(device=gate.weight.device, dtype=torch.float32)
        if weight.shape != gate.weight.shape:
            raise ValueError(f"{name}: checkpoint shape {tuple(weight.shape)} differs from the loaded gate")

        def forward(x, weight=weight):
            return F.linear(x.float(), weight), None

        patched[number] = gate.forward
        gate.forward = forward
    if not patched:
        raise ValueError("No MoE router gate found to run in float32")
    return patched
