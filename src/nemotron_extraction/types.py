from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
import torch

KINDS = ("resid_pre", "resid_post", "mamba_out", "moe_out", "attn_out", "attn_out_prelinear")

@dataclass(frozen=True)
class HookSite:
    """A semantic activation site; layer indices are zero-based."""
    layer: int
    kind: str

    def __post_init__(self) -> None:
        if isinstance(self.layer, bool) or not isinstance(self.layer, int) or self.layer < 0:
            raise ValueError("layer must be a nonnegative integer")
        if self.kind not in KINDS:
            raise ValueError(f"Unsupported hook kind {self.kind!r}; choose from {KINDS}")

    @property
    def key(self) -> str:
        return f"L{self.layer}_{self.kind}"

def validate_sites(sites: Iterable[HookSite]) -> tuple[HookSite, ...]:
    result = tuple(sites)
    if not result or len(set(result)) != len(result):
        raise ValueError("Provide at least one site, without duplicates")
    return result

def validate_publication_threads(threads: int) -> int:
    if type(threads) is not int or threads not in (1, 2):
        raise ValueError("publication_threads must be 1 or 2")
    return threads

def validate_tokens(token_ids: torch.Tensor) -> torch.Tensor:
    if token_ids.dtype != torch.long or token_ids.ndim != 2 or not all(token_ids.shape):
        raise ValueError("token_ids must be a nonempty rectangular int64 [batch, length] tensor")
    if bool((token_ids < 0).any()):
        raise ValueError("token_ids must be nonnegative")
    return token_ids.detach().cpu().clone()

@dataclass
class ActivationBatch:
    """CPU token IDs and sequence-major, unpadded [batch * length, width] rows."""
    token_ids: torch.Tensor
    activations: dict[str, torch.Tensor]

    def __post_init__(self) -> None:
        if self.token_ids.ndim != 2 or self.token_ids.device.type != "cpu":
            raise ValueError("ActivationBatch requires CPU [batch, length] token IDs")
        for key, value in self.activations.items():
            if value.ndim != 2 or value.shape[0] != self.token_ids.numel() or value.device.type != "cpu":
                raise ValueError(f"{key}: expected CPU [{self.token_ids.numel()}, width] rows")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"{key}: activations contain nonfinite values")


@dataclass
class DeviceActivationBatch:
    """CPU token IDs with detached device-resident rows for live SAE training."""
    token_ids: torch.Tensor
    activations: dict[str, torch.Tensor]

    def __post_init__(self):
        if self.token_ids.ndim != 2 or self.token_ids.device.type != "cpu":
            raise ValueError("DeviceActivationBatch requires CPU token IDs")
        for key, value in self.activations.items():
            if value.ndim != 2 or value.shape[0] != self.token_ids.numel() or value.device.type != "cuda":
                raise ValueError(f"{key}: expected CUDA [tokens, width] activations")
