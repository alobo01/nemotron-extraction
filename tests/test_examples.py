import runpy
import sys
from pathlib import Path


def test_toy_extraction_example_runs(tmp_path, monkeypatch, capsys):
    example = Path(__file__).resolve().parents[1] / "examples" / "toy_extraction.py"
    monkeypatch.setattr(sys, "argv", [str(example), str(tmp_path / "out")])
    runpy.run_path(str(example), run_name="__main__")
    output = capsys.readouterr().out
    assert "L2_resid_post            (10, 8)" in output
    assert sorted(path.name for path in (tmp_path / "out").iterdir()) == ["000000", "000001"]
