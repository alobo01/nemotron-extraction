"""Pinned inputs, provenance and JSON receipts shared by every benchmark."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

import nemotron_extraction
from nemotron_extraction.transport import atomic_path

MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
MODEL_REVISION = "fe4143e388795a58a92974891b2e45092a53dc1d"
DATA = {
    "en": ("HuggingFaceFW/fineweb-edu", "sample-10BT"),
    "de": ("HuggingFaceFW/fineweb-2", "deu_Latn"),
    "es": ("HuggingFaceFW/fineweb-2", "spa_Latn"),
}
SEQUENCE_LENGTH = 2048
# First frozen English sequence, as little-endian int64 token IDs.
ANCHOR_SHA256 = "c5e6f8311596eba310ae9c0f181334dc5d2ae8fcbd8290d7f26ce995b30fb840"


def digest(path):
    """Stream the file so multi-gigabyte shards are never read into memory at once."""
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def tensor_digest(value):
    return hashlib.sha256(value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def write_json(path, value):
    with atomic_path(path) as partial:
        partial.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def versions():
    result = {"python": sys.version.split()[0], "cuda": torch.version.cuda}
    for name in (
        "nemotron-extraction",
        "torch",
        "transformers",
        "vllm",
        "datasets",
        "safetensors",
        "mamba-ssm",
        "causal-conv1d",
        "accelerate",
        "kernels",
    ):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def package_hashes():
    """Source hashes of the imported extraction package, which may be an installed wheel."""
    return {path.name: digest(path) for path in sorted(Path(nemotron_extraction.__file__).parent.glob("*.py"))}


def load_tokens(work, language, manifest=None):
    """Frozen token IDs for one language, refused if the file changed since `prepare`."""
    manifest = manifest or json.loads((Path(work) / "manifest.json").read_text())
    path = Path(work) / f"{language}.safetensors"
    if digest(path) != manifest["languages"][language]["token_sha256"]:
        raise ValueError(f"Frozen {language} token checksum differs from the manifest")
    return load_file(str(path))["token_ids"]


def batches(tokens, batch_size, seq_len):
    """Full [batch, seq_len] rows of a flat token stream, then one shorter final row."""
    full = tokens.numel() // seq_len * seq_len
    if full:
        yield from tokens[:full].reshape(-1, seq_len).split(batch_size)
    if full < tokens.numel():
        yield tokens[full:].unsqueeze(0)


def anchor_inputs(work):
    """The pinned manifest and the first 64 frozen English sequences."""
    manifest = json.loads((Path(work) / "manifest.json").read_text())
    if manifest["model"] != MODEL or manifest["model_revision"] != MODEL_REVISION or manifest["dtype"] != "bfloat16":
        raise ValueError("Expected the pinned official BF16 checkpoint")
    tokens = load_tokens(work, "en", manifest)
    if tokens.numel() < 64 * SEQUENCE_LENGTH:
        raise ValueError("Need the first 64 frozen English sequences")
    return manifest, tokens[: 64 * SEQUENCE_LENGTH].reshape(64, SEQUENCE_LENGTH)


def anchor_digest(rows):
    return hashlib.sha256(rows[0].cpu().numpy().astype("<i8", copy=False).tobytes()).hexdigest()


def is_memory_failure(error):
    message = str(error).lower()
    return isinstance(error, torch.OutOfMemoryError) or any(
        text in message for text in ("out of memory", "gpu memory budget", "no available memory", "kv cache blocks")
    )


def job_minutes(seconds):
    """Cluster time request: 25% margin plus two minutes, rounded up to five-minute steps."""
    return math.ceil((seconds * 1.25 + 120) / 300) * 5
