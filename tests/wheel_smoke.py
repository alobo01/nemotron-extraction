"""Run from a temporary cwd with only the installed wheel target on PYTHONPATH."""
import importlib
import importlib.metadata
import inspect

package = importlib.import_module("nemotron_extraction")
worker = importlib.import_module("nemotron_extraction.worker")
assert worker.ActivationHookExtension
assert "wheel-check" in str(package.__file__)
assert importlib.metadata.version("nemotron-extraction") == package.__version__
from nemotron_extraction import HFExtractor, VLLMExtractor
for extractor in (HFExtractor,VLLMExtractor):
    assert "diagnostics" in inspect.signature(extractor).parameters
    assert "publication_threads" in inspect.signature(extractor).parameters
print(f"Installed wheel {package.__version__}; worker extension imports outside checkout")
