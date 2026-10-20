import sys
from pathlib import Path

# Benchmark scripts and their benchlib helpers are importable by module name, as when run.
sys.path.insert(0, str(Path(__file__).parents[1] / "benchmarks"))
