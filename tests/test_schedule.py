import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("schedule", Path(__file__).parents[1] / "benchmarks/schedule.py")
schedule = importlib.util.module_from_spec(spec)
spec.loader.exec_module(schedule)


def test_long_runs_split_into_contiguous_bounded_ranges(tmp_path):
    for language in ("en", "de", "es"):
        root = tmp_path / f"{language}-0"
        root.mkdir()
        values = {"tokens": 16384, "batch_size": 4, "startup_seconds": 60,
                  "extraction_seconds": 60, "persistence_seconds": 5,
                  "comparison_seconds": 10, "wall_seconds": 140}
        for backend in ("hf", "vllm"):
            (root / f"{backend}.json").write_text(json.dumps(values))
    jobs = schedule.schedule(tmp_path, 4)
    for language in ("en", "de", "es"):
        selected = [job for job in jobs if job["language"] == language]
        assert len(selected) > 1
        assert sum(job["tokens"] for job in selected) == 1_000_000
        end = 0
        for job in selected:
            assert job["start"] == end and end % 2048 == 0
            assert job["minutes"] <= 60 and job["minutes"] % 5 == 0
            end += job["tokens"]
    with pytest.raises(ValueError, match="batch size"):
        schedule.schedule(tmp_path, 2)


def test_all_languages_budget_for_cold_start(tmp_path):
    for language in ("en", "de", "es"):
        root = tmp_path / f"{language}-0"
        root.mkdir()
        for backend in ("hf", "vllm"):
            startup = (600 if backend == "hf" else 60) if language == "en" else 1
            value = {"tokens": 16384, "batch_size": 4, "startup_seconds": startup,
                     "extraction_seconds": 1, "persistence_seconds": 1,
                     "comparison_seconds": 1, "wall_seconds": startup + 2}
            (root / f"{backend}.json").write_text(json.dumps(value))
    jobs = schedule.schedule(tmp_path, 4)
    assert len(jobs) == 3
    assert {job["minutes"] for job in jobs} == {25}
