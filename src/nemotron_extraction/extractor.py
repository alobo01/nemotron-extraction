"""State and metrics shared by the PyTorch and vLLM extractors."""

from __future__ import annotations

from pathlib import Path

from .transport import MEMORY_KEYS
from .types import validate_publication_threads, validate_sites


class Extractor:
    def __init__(self, sites, diagnostics, publication_threads):
        self.sites = validate_sites(sites)
        self.diagnostics = diagnostics
        self.publication_threads = validate_publication_threads(publication_threads)
        self.closed = False
        self.actual_forward_batches = []
        self.validation_seconds = self.serialization_seconds = self.worker_write_seconds = 0.0
        self.backpressure_seconds = self.final_drain_seconds = self.driver_read_seconds = 0.0
        for key in MEMORY_KEYS:
            setattr(self, key, 0)

    def _require_open(self):
        if self.closed:
            raise RuntimeError("Extractor is closed")

    def _fold_memory(self, peaks):
        for key in MEMORY_KEYS:
            setattr(self, key, max(getattr(self, key), peaks[key]))

    @staticmethod
    def shard_path(output_dir, index):
        return Path(output_dir) / f"{index:06d}.safetensors"

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
