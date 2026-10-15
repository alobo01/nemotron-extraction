"""Bounded pinned buffers and event-ordered GPU-to-host transfers."""
from __future__ import annotations

import threading
import time
from pathlib import Path

import torch
from safetensors.torch import save_file


class BufferPool:
    """Two slots; release only after a consumer finishes."""
    def __init__(self):
        self.slots = [{}, {}]
        self.free = [0, 1]
        self.lock = threading.Lock()

    def acquire(self):
        with self.lock:
            return self.free.pop(0) if self.free else None

    def tensor(self, slot, key, shape, dtype, *, pinned):
        buffers = self.slots[slot]
        tensor = buffers.get(key)
        if tensor is None or tuple(tensor.shape) != tuple(shape) or tensor.dtype != dtype:
            tensor = torch.empty(shape, dtype=dtype, pin_memory=pinned)
            buffers[key] = tensor
        return tensor

    def release(self, slot):
        with self.lock:
            if slot in self.free:
                raise RuntimeError("Buffer slot released twice")
            self.free.append(slot)


def finish_copy(event, started, ended, sources):
    """Runs on a host thread, never the inference worker's RPC thread."""
    if event is not None:
        event.synchronize()
        seconds = started.elapsed_time(ended) / 1000
    else:
        seconds = 0.0
    sources.clear()
    return seconds


def publish(tensors, path, metadata=None):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    started = time.perf_counter()
    try:
        for key, tensor in tensors.items():
            if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"{key}: nonfinite activations")
        save_file(tensors, str(partial), metadata=metadata)
        partial.replace(path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return {"path": str(path), "bytes": path.stat().st_size,
            "write_seconds": time.perf_counter() - started}
