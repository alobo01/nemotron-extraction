"""Run from a temporary cwd with only the installed wheel target on PYTHONPATH."""
import importlib
import importlib.metadata
import inspect

package = importlib.import_module("nemotron_extraction")
worker = importlib.import_module("nemotron_extraction.worker")
assert worker.ActivationHookExtension
assert "wheel-check" in str(package.__file__)
assert importlib.metadata.version("nemotron-extraction") == "1.0.0"
from nemotron_extraction import HFExtractor, VLLMExtractor
from nemotron_extraction.diagnostics import DiagnosticWorker, IntegrityWorker
from nemotron_extraction.precision import PrecisionWorker
assert DiagnosticWorker.diag_install and IntegrityWorker.nh_install
assert PrecisionWorker.precision_prepare and PrecisionWorker.precision_case
for extractor in (HFExtractor,VLLMExtractor):
    assert "diagnostics" in inspect.signature(extractor).parameters
    assert "publication_threads" in inspect.signature(extractor).parameters
print(f"Installed wheel {package.__version__}; worker extension imports outside checkout")
