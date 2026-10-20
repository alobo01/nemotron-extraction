"""Verify installed worker discovery and engine interfaces outside the checkout."""
import importlib.metadata
import inspect
from pathlib import Path
import nemotron_extraction
from nemotron_extraction.worker import ActivationHookExtension
from nemotron_extraction.diagnostics import DiagnosticWorker, IntegrityWorker
assert DiagnosticWorker.diag_install and IntegrityWorker.nh_install

print("package:", Path(nemotron_extraction.__file__).resolve())
print("worker:", ActivationHookExtension.__module__)
from vllm import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.engine.input_processor import InputProcessor

print("vllm:", importlib.metadata.version("vllm"))
for name in ("collective_rpc", "generate", "shutdown", "abort"):
    print(name, inspect.signature(getattr(AsyncLLM, name)))
for name in ("max_num_seqs", "kv_cache_memory_bytes", "worker_extension_cls", "enforce_eager"):
    assert name in inspect.signature(AsyncEngineArgs).parameters, name
source = inspect.getsource(GPUModelRunner)
for fragment in ("self.input_batch", "self.query_start_loc", "num_computed_tokens_cpu"):
    assert fragment in source, f"Unsupported runner: missing {fragment}"
print("request-ID assignment:")
print(inspect.getsource(InputProcessor.assign_request_id))
print("Installed package and worker interfaces verified")
