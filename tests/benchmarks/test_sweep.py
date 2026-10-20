import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
import torch
from safetensors.torch import save_file
from nemotron_extraction import HFExtractor, HookSite
from tiny_models import Model

import sweep
from benchlib.common import digest, write_json

def test_durable_shard_resume_and_comparison(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "CPU test fixture")
    work, output = tmp_path / "tokens", tmp_path / "output"
    work.mkdir()
    path = work / "en.safetensors"
    save_file({"token_ids": torch.arange(6)}, str(path))
    manifest = {"model_revision": "fixed", "tokenizer_revision": "fixed",
                "languages": {"en": {"token_sha256": digest(path)}}}
    write_json(work / "manifest.json", manifest)
    args = argparse.Namespace(work=work, output=output, backend="hf", batch_size=8, tokens=6, profile=True)
    with HFExtractor(Model(), [HookSite(0, "resid_post")], device="cpu") as extractor:
        record = asyncio.run(sweep.language(extractor, args, manifest, "en", None))
        assert record["tokens"] == 6 and record["status"] == "complete"
        run = output / "b8" / "en"
        (run / "hf.json").unlink()  # Interrupted after a durable shard, before the language receipt.
        resumed = asyncio.run(sweep.language(extractor, args, manifest, "en", None))
        assert resumed["resumed_shards"] == 1 and resumed["tokens"] == 6
    write_json(output / "b8" / "hf-run.json", {"status": "complete"})
    args.backend = "vllm"
    with HFExtractor(Model(), [HookSite(0, "resid_post")], device="cpu") as extractor, ThreadPoolExecutor(1) as pool:
        candidate = asyncio.run(sweep.language(extractor, args, manifest, "en", pool))
    assert candidate["passed"] and candidate["agreement"]["L0_resid_post"]["tokens"] == 6
    assert not list((run / "reference").glob("*.safetensors"))
    assert (run / "checkpoints" / "vllm-000000.json").exists()
    assert (run / "checkpoints" / "vllm-000000.safetensors").exists()
    assert (run / "vllm.json").exists()
