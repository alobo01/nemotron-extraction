import sys
from pathlib import Path

tests = Path(__file__).parent
# Benchmark scripts and benchlib import by module name, as when the scripts run; tiny_models is shared.
sys.path[:0] = [str(tests.parent / "benchmarks"), str(tests)]
