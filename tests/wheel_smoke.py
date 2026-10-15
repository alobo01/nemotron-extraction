"""Run from a temporary cwd with only the installed wheel target on PYTHONPATH."""
import importlib
import importlib.metadata

package = importlib.import_module("nemotron_extraction")
worker = importlib.import_module("nemotron_extraction.worker")
assert worker.ActivationHookExtension
assert "wheel-check" in str(package.__file__)
assert importlib.metadata.version("nemotron-extraction") == "1.0.0"
print(f"Installed wheel {package.__version__}; worker extension imports outside checkout")
