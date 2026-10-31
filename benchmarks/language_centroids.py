"""Per-language mean block outputs of a decoder, summed on the GPU by `AggregateExtractor`.

Reads a JSONL manifest with `language_nllb`, `example_id` and `text`, and writes
`centroids.safetensors` (`sum_`, `count_` and `mean_` per language, each layer by width) and
`runtime.json`. Nothing but the sums leaves the device, so there is no copy or disk traffic to time.

    python benchmarks/language_centroids.py --model MODEL --manifest rows.jsonl --output OUT \
        --num-layers 60 --tensor-parallel-size 4 --max-model-len 512
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import torch
from benchlib.common import digest, package_hashes, versions, write_json
from safetensors.torch import save_file
from transformers import AutoTokenizer

from nemotron_extraction.aggregate import AggregateExtractor


def read_manifest(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("Manifest is empty")
    for number, row in enumerate(rows, 1):
        for key in ("language_nllb", "text"):
            if key not in row:
                raise ValueError(f"Manifest line {number} misses {key}")
    return rows


async def run(args):
    rows = read_manifest(args.manifest)[: args.limit]
    languages = sorted({row["language_nllb"] for row in rows})
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, local_files_only=True)
    started = time.perf_counter()
    encoded = tokenizer([row["text"] for row in rows], add_special_tokens=True, truncation=False)["input_ids"]
    tokenize_seconds = time.perf_counter() - started
    too_long = [len(ids) for ids in encoded if len(ids) > args.max_model_len]
    if too_long:
        raise ValueError(f"{len(too_long)} rows exceed max_model_len={args.max_model_len}")
    group = {language: number for number, language in enumerate(languages)}
    requests = [(f"r{n}", group[row["language_nllb"]], ids) for n, (row, ids) in enumerate(zip(rows, encoded))]
    async with AggregateExtractor(
        args.model,
        list(range(args.num_layers)),
        len(languages),
        revision=args.revision,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.token_budget,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        engine_options=json.loads(args.engine_options),
    ) as extractor:
        sums, counts = await extractor.extract(requests)
    tokens = int(counts.sum())
    tensors = {}
    for index, language in enumerate(languages):
        tensors[f"sum_{language}"] = sums[:, index].float().contiguous()
        tensors[f"count_{language}"] = counts[index].clone()
        tensors[f"mean_{language}"] = (sums[:, index] / int(counts[index])).float().contiguous()
    args.output.mkdir(parents=True, exist_ok=True)
    save_file(
        tensors,
        str(args.output / "centroids.safetensors"),
        metadata={
            "backend": "nemotron-extraction-aggregate",
            "dtype": "bfloat16",
            "pooling": "global_token_mean",
            "layers": str(args.num_layers),
            "hidden_size": str(sums.shape[-1]),
        },
    )
    seconds = extractor.extract_seconds
    write_json(
        args.output / "runtime.json",
        {
            "backend": "nemotron-extraction-aggregate",
            "model": args.model,
            "revision": args.revision,
            "request_count": len(requests),
            "non_padding_tokens": tokens,
            "language_token_counts": {language: int(counts[i]) for i, language in enumerate(languages)},
            "engine_startup_seconds": extractor.engine_startup_seconds,
            "hook_install_seconds": extractor.hook_install_seconds,
            "tokenize_seconds": tokenize_seconds,
            "extraction_seconds": seconds,
            "requests_per_second": len(requests) / seconds,
            "non_padding_tokens_per_second": tokens / seconds,
            "tensor_parallel_size": args.tensor_parallel_size,
            "token_budget": args.token_budget,
            "max_num_seqs": args.max_num_seqs,
            "max_model_len": args.max_model_len,
            "manifest_sha256": digest(args.manifest),
            "aggregate_only": True,
            "versions": versions(),
            "package_hashes": package_hashes(),
        },
    )
    print(f"{tokens} tokens in {seconds:.1f} s = {tokens / seconds:.0f} tokens/s")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-layers", type=int, required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--token-budget", type=int, default=16384)
    parser.add_argument("--max-num-seqs", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--engine-options", default="{}", help="JSON of further vLLM engine arguments")
    parser.add_argument("--limit", type=int, help="Use only the first N manifest rows")
    args = parser.parse_args()
    if (args.output / "centroids.safetensors").exists():
        raise FileExistsError(f"Refusing to overwrite {args.output / 'centroids.safetensors'}")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
