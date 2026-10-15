"""Estimate tight job bounds from the three completed pilot receipts.

Prints a sequential job schedule; it does not submit jobs.
uv run --no-project benchmarks/schedule.py --pilot PATH --batch-size 4
"""
import argparse
import json
import math
from pathlib import Path


def schedule(pilot, batch_size):
    jobs = []
    records = {}
    for language in ("en", "de", "es"):
        path = pilot / f"{language}-0"
        hf = json.loads((path / "hf.json").read_text())
        vllm = json.loads((path / "vllm.json").read_text())
        if hf["tokens"] != 16384 or vllm["tokens"] != 16384:
            raise ValueError("Need complete 16,384-token pilot receipts")
        if hf["batch_size"] != batch_size or vllm["batch_size"] != batch_size:
            raise ValueError("Both pilot backends must use the selected batch size")
        records[language] = (hf, vllm)
    # A new allocation can have cold weights even if later pilot languages were warm.
    startup = max(pair[0]["startup_seconds"] for pair in records.values())
    startup += max(pair[1]["startup_seconds"] for pair in records.values())
    for language, (hf, vllm) in records.items():
        # Wall minus startup includes warmup, cleanup, persistence and comparison.
        fixed = max(0, hf["wall_seconds"] - hf["startup_seconds"] - hf["extraction_seconds"] - hf["persistence_seconds"])
        fixed += max(0, vllm["wall_seconds"] - vllm["startup_seconds"] - vllm["extraction_seconds"] - vllm["comparison_seconds"])
        per_token = (hf["extraction_seconds"] + hf["persistence_seconds"] +
                     vllm["extraction_seconds"] + vllm["comparison_seconds"]) / 16384
        available = (3600 - 120) / 1.25 - startup - fixed
        maximum = int(available / per_token / 2048) * 2048
        if maximum < 2048:
            raise ValueError("Model startup leaves no useful work in a one-hour job")
        start = 0
        while start < 1_000_000:
            count = min(maximum, 1_000_000 - start)
            minutes = math.ceil((1.25 * (startup + fixed + count * per_token) + 120) / 300) * 5
            jobs.append({"language": language, "start": start, "tokens": count,
                         "minutes": min(60, minutes), "batch_size": batch_size})
            start += count
    return jobs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, choices=(1, 2, 4, 8, 16, 32, 64, 128), default=4)
    args = parser.parse_args()
    print(json.dumps(schedule(args.pilot, args.batch_size), indent=2))
