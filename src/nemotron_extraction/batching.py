"""Fit extraction waves to device, pinned-memory and publication byte budgets."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class AutoBatchConfig:
    resident_bytes: int
    reference_peak_bytes: int
    reference_sequences: int
    activation_bytes_per_token: int
    sequence_length: int = 2048
    frame_sequences: int = 4
    host_capture_budget_bytes: int = 32 * 1024**3
    publication_budget_bytes: int = 24 * 1024**3
    workspace_reserve_bytes: int = 2 * 1024**3

    def fit(self, total_bytes, fraction=0.96, *, external_used_bytes=0):
        if (
            not 0 < fraction <= 1
            or min(
                self.reference_sequences, self.sequence_length, self.frame_sequences, self.activation_bytes_per_token
            )
            <= 0
        ):
            raise ValueError("Invalid automatic batch geometry or memory fraction")
        per_sequence = max(
            (self.reference_peak_bytes - self.resident_bytes) / self.reference_sequences,
            self.activation_bytes_per_token * self.sequence_length,
        )
        available = fraction * total_bytes - external_used_bytes - self.resident_bytes - self.workspace_reserve_bytes
        device = math.floor(available / per_sequence)
        payload = self.activation_bytes_per_token * self.sequence_length
        host = self.host_capture_budget_bytes // (2 * payload)
        publication = self.publication_budget_bytes // payload
        result = int(min(device, host, publication)) // self.frame_sequences * self.frame_sequences
        if result < self.frame_sequences:
            raise MemoryError("Not enough bounded device/host/publication memory for one archive frame")
        return result
