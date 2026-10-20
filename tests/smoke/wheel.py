"""Check the installed wheel from outside the checkout: python tests/smoke/wheel.py, run from another directory."""
import importlib.metadata
import inspect
from pathlib import Path

import nemotron_extraction
from nemotron_extraction import HFExtractor, VLLMExtractor
from nemotron_extraction.worker import ActivationHookExtension

checkout = Path(__file__).resolve().parents[2] / "src"
assert checkout not in Path(nemotron_extraction.__file__).resolve().parents, "Imported the checkout, not the wheel"
assert importlib.metadata.version("nemotron-extraction") == nemotron_extraction.__version__
assert ActivationHookExtension.nh_install
for extractor in (HFExtractor, VLLMExtractor):
    parameters = inspect.signature(extractor).parameters
    assert "diagnostics" in parameters and "publication_threads" in parameters
print(f"Installed wheel {nemotron_extraction.__version__} imports outside the checkout")
