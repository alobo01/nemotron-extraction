"""GPU utilisation sampling and timeline overlap for throughput receipts."""
from __future__ import annotations

import json
import subprocess
import threading
import time

import torch


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


def chrome_trace(timeline, process):
    """Timeline events in the Chrome trace format, viewable in chrome://tracing or Perfetto."""
    return {"traceEvents": [
        {"name": item["name"], "ph": "X", "pid": process, "tid": item["name"],
         "ts": item["start"] * 1e6, "dur": (item["end"] - item["start"]) * 1e6,
         "args": {"attempt": item.get("attempt")}} for item in timeline]}
