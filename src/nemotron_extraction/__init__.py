"""Token-aligned activation extraction; backend imports are lazy."""
from .types import ActivationBatch, HookSite
from .hf import HFExtractor
from .vllm import VLLMExtractor

__all__ = ["ActivationBatch", "HookSite", "HFExtractor", "VLLMExtractor"]
__version__ = "1.0.0"
