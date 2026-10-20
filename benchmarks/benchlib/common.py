"""Hashing and JSON receipts shared by every benchmark."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from nemotron_extraction.transport import atomic_path


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
