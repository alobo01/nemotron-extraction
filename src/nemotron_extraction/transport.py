"""Bounded pinned buffers and event-ordered GPU-to-host transfers."""
from __future__ import annotations

import threading
import time
from pathlib import Path

import torch
from safetensors.torch import save_file

MEMORY_KEYS = ("peak_gpu_memory_bytes", "peak_gpu_reserved_bytes", "peak_device_memory_bytes",
               "total_device_memory_bytes", "peak_device_memory_fraction")
# One capture being filled while the previous one is written.
SLOTS = 2


class BufferPool:
    """Release a slot only after its consumer finishes."""
    def __init__(self):
        self.slots = [{} for _ in range(SLOTS)]
        self.free = list(range(SLOTS))
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

    def reset(self):
        """Only call once transfers and writers have drained; no slot can still be read."""
        with self.lock:
            self.free = list(range(SLOTS))
            for slot in self.slots:
                slot.clear()


def seconds(started, ended):
    return started.elapsed_time(ended) / 1000


def finish_copy(started, ended, sources):
    """Runs on a host thread, never the inference worker's RPC thread."""
    elapsed = 0.0
    if ended is not None:
        ended.synchronize()
        elapsed = seconds(started, ended)
    sources.clear()
    return elapsed


def capture_seconds(events):
    return sum(seconds(started, ended) for _, started, ended in events)


def timeline(origin, spans):
    """Place named (start, end) CUDA event pairs relative to a shared origin event."""
    return [{"name": name, "start": seconds(origin, started), "end": seconds(origin, ended)}
            for name, pairs in spans for started, ended in pairs]


def memory_peaks(previous=None, device=None):
    """Fold current CUDA memory statistics into running maxima keyed by MEMORY_KEYS."""
    args = () if device is None else (device,)
    free, total = torch.cuda.mem_get_info(*args)
    current = dict(zip(MEMORY_KEYS, (
        torch.cuda.max_memory_allocated(*args), torch.cuda.max_memory_reserved(*args),
        total - free, total, (total - free) / total if total else 0.0)))
    previous = previous or {}
    return {key: max(value, previous.get(key, 0)) for key, value in current.items()}


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
        validated = time.perf_counter()
        save_file(tensors, str(partial), metadata=metadata)
        serialized = time.perf_counter()
        partial.replace(path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    finished = time.perf_counter()
    return {"path": str(path), "bytes": path.stat().st_size,
            "validation_seconds": validated - started,
            "serialization_seconds": serialized - validated,
            "publication_seconds": finished - serialized,
            "write_seconds": finished - started}
