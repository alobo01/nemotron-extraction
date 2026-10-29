"""Bounded pinned buffers and event-ordered GPU-to-host transfers."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

import torch
from safetensors.torch import save_file

MEMORY_KEYS = (
    "peak_gpu_memory_bytes",
    "peak_gpu_reserved_bytes",
    "peak_device_memory_bytes",
    "total_device_memory_bytes",
    "peak_device_memory_fraction",
)
# By default one capture is filled while the previous one is written.
SLOTS = 2


class BufferPool:
    """Release a slot only after its consumer finishes."""

    def __init__(self, slots=SLOTS):
        if type(slots) is not int or slots < 2:
            raise ValueError("A buffer pool needs at least two slots")
        self.slots = [{} for _ in range(slots)]
        self.free = list(range(slots))
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
            self.free = list(range(len(self.slots)))
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
    return [
        {"name": name, "start": seconds(origin, started), "end": seconds(origin, ended)}
        for name, pairs in spans
        for started, ended in pairs
    ]


def memory_peaks(previous=None, device=None):
    """Fold current CUDA memory statistics into running maxima keyed by MEMORY_KEYS."""
    args = () if device is None else (device,)
    free, total = torch.cuda.mem_get_info(*args)
    current = dict(
        zip(
            MEMORY_KEYS,
            (
                torch.cuda.max_memory_allocated(*args),
                torch.cuda.max_memory_reserved(*args),
                total - free,
                total,
                (total - free) / total if total else 0.0,
            ),
        )
    )
    previous = previous or {}
    return {key: max(value, previous.get(key, 0)) for key, value in current.items()}


def fsync_file(path):
    with open(path, "r+b") as handle:
        os.fsync(handle.fileno())


def fsync_directory(path):
    """Persist directory entries; Windows has no directory handles to sync."""
    if os.name == "nt":
        return
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def device_finite(source):
    """One device-side flag; chunks bound the temporary boolean tensor."""
    return torch.stack([torch.isfinite(chunk).all() for chunk in source.split(8192)]).all()


@contextmanager
def atomic_path(path):
    """Yield a temporary sibling path that durably replaces ``path`` only if the block succeeds."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    try:
        yield partial
        fsync_file(partial)
        partial.replace(path)
        fsync_directory(path.parent)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def publish(tensors, path, metadata=None):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    started = time.perf_counter()
    with atomic_path(path) as partial:
        for key, tensor in tensors.items():
            if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"{key}: nonfinite activations")
        validated = time.perf_counter()
        save_file(tensors, str(partial), metadata=metadata)
        serialized = time.perf_counter()
    finished = time.perf_counter()
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "validation_seconds": validated - started,
        "serialization_seconds": serialized - validated,
        "publication_seconds": finished - serialized,
        "write_seconds": finished - started,
    }


def tensor_sha256(tensor):
    return hashlib.sha256(memoryview(tensor.contiguous().view(torch.uint8).numpy())).hexdigest()


def publish_site_directory(tensors, path, writer, metadata=None):
    """Publish immutable site files together, with hashes from completed host buffers.

    The caller must have checked floating-point values before submitting this
    operation. The vLLM per-site path does so on the device before its copy ends.
    """
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    partial = path.with_name(path.name + "." + uuid.uuid4().hex + ".partial")
    partial.mkdir(parents=True)
    started = time.perf_counter()

    def write(key, tensor):
        if not key or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_" for c in key):
            raise ValueError("Unsafe tensor file key")
        file = partial / (key + ".safetensors")
        digest = tensor_sha256(tensor)
        save_file({key: tensor}, str(file), metadata=metadata)
        fsync_file(file)
        return key, dict(
            path=str(path / file.name),
            sha256=digest,
            shape=list(tensor.shape),
            dtype=str(tensor.dtype),
            bytes=file.stat().st_size,
        )

    try:
        futures = [writer.submit(write, key, tensor) for key, tensor in tensors.items()]
        files = {}
        error = None
        for future in futures:
            try:
                key, value = future.result()
                files[key] = value
            except BaseException as exc:
                error = error or exc
        if error is not None:
            raise error
        manifest = dict(version=2, layout="per_site", files=files, finite_validation="device")
        with (partial / "receipt.json").open("x") as handle:
            json.dump(manifest, handle)
            handle.flush()
            os.fsync(handle.fileno())
        total = sum(p.stat().st_size for p in partial.iterdir())
        # Publish only after every site file and receipt is complete.
        fsync_directory(partial)
        os.rename(partial, path)
        fsync_directory(path.parent)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    finished = time.perf_counter()
    return dict(
        manifest,
        path=str(path),
        bytes=total,
        validation_seconds=0.0,
        serialization_seconds=finished - started,
        publication_seconds=0.0,
        write_seconds=finished - started,
    )
