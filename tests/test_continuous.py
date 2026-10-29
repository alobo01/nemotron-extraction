"""The vLLM driver and worker together under chunked prefill and concurrent capture waves."""

import asyncio
import sys
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file
from test_worker import resid_post, schedule, worker  # noqa: F401  (fixture)

from nemotron_extraction import HookSite, VLLMExtractor


class ChunkingEngine:
    """Admit requests first-come up to ``max_seqs`` and spend ``budget`` prompt tokens per forward."""

    def __init__(self, extension, budget, max_seqs, fail=None):
        self.worker, self.budget, self.max_seqs, self.fail = extension, budget, max_seqs, fail
        self.queue, self.forwards, self.task = [], [], None

    async def generate(self, inputs, params, rid):
        if rid == self.fail:
            raise RuntimeError("engine failure")
        done = asyncio.get_running_loop().create_future()
        self.queue.append(dict(rid=rid, prompt=inputs["prompt_token_ids"], computed=0, done=done))
        if self.task is None or self.task.done():
            self.task = asyncio.get_running_loop().create_task(self.run())
        await done
        yield None

    async def run(self):
        while self.queue:
            await asyncio.sleep(0)  # let the driver admit and submit further waves
            entries, left = [], self.budget
            for request in self.queue[: self.max_seqs]:
                count = min(left, len(request["prompt"]) - request["computed"])
                if count:
                    entries.append((request, count))
                    left -= count
            # vLLM appends an internal suffix to every request ID.
            schedule(self.worker, [(r["rid"] + "-0a1b2c3d", r["prompt"], r["computed"], n) for r, n in entries])
            self.forwards.append({r["rid"].rsplit("-", 1)[0] for r, _ in entries})
            for request, count in entries:
                request["computed"] += count
                if request["computed"] == len(request["prompt"]):
                    self.queue.remove(request)
                    if not request["done"].done():  # a cancelled caller is aborted afterwards
                        request["done"].set_result(None)

    async def abort(self, rid):
        self.queue = [request for request in self.queue if request["rid"] != rid]

    async def collective_rpc(self, method, args=()):
        if method == "nh_install":
            self.worker.nh_remove()  # reinstall with the driver's settings
        return [getattr(self.worker, method)(*args)]

    def shutdown(self):
        pass


@pytest.fixture
def engine(worker, monkeypatch):  # noqa: F811
    made = []

    def build(budget, max_seqs, fail=None):
        def from_engine_args(args):
            made.append(ChunkingEngine(worker, budget, max_seqs, fail))
            made[-1].args = vars(args)
            return made[-1]

        modules = dict(AsyncEngineArgs=lambda **k: SimpleNamespace(**k), SamplingParams=lambda **k: k)
        monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(**modules))
        monkeypatch.setitem(
            sys.modules,
            "vllm.v1.engine.async_llm",
            SimpleNamespace(AsyncLLM=SimpleNamespace(from_engine_args=from_engine_args)),
        )
        return made

    return build


def extract(tmp_path, batches, **options):
    sites = [HookSite(0, "resid_pre"), HookSite(0, "resid_post")]

    async def run(extractor):
        return [receipt async for receipt in extractor.stream_to_disk(batches, tmp_path / "out")]

    with VLLMExtractor("fixture", sites, max_model_len=8, sink_dir=tmp_path, **options) as extractor:
        return asyncio.run(run(extractor)), extractor


def waves(count, rows=2, length=5):
    return [torch.arange(i * 100, i * 100 + rows * length).reshape(rows, length) for i in range(count)]


def check(receipts, batches, per_site):
    assert [receipt["index"] for receipt in receipts] == list(range(len(batches)))
    for receipt, batch in zip(receipts, batches):
        if per_site:
            saved = load_file(receipt["files"]["L0_resid_post"]["path"])
        else:
            saved = load_file(receipt["path"])
            assert torch.equal(saved["token_ids"], batch)
        assert saved["L0_resid_post"].flatten().tolist() == resid_post(batch.flatten().tolist())


@pytest.mark.parametrize("layout", ["monolithic", "per_site"])
def test_concurrent_chunked_waves_share_forwards_and_keep_rows(engine, tmp_path, layout):
    made = engine(budget=3, max_seqs=4)
    batches = waves(4)
    receipts, extractor = extract(
        tmp_path,
        batches,
        sequences_per_call=2,
        enable_chunked_prefill=True,
        max_num_batched_tokens=3,
        concurrent_waves=True,
        capture_slots=3,
        publication_layout=layout,
    )
    check(receipts, batches, layout == "per_site")
    assert made[0].args["max_num_batched_tokens"] == 3
    assert any(len(forward) > 1 for forward in made[0].forwards), "no forward mixed two waves"
    assert sum(receipts[0]["actual_forward_batches"]) >= 4  # each 5-token prompt spans forwards
    assert not extractor.failed


def test_serial_waves_never_share_a_forward(engine, tmp_path):
    made = engine(budget=3, max_seqs=4)
    batches = waves(3)
    receipts, _ = extract(
        tmp_path, batches, sequences_per_call=2, enable_chunked_prefill=True, max_num_batched_tokens=3
    )
    check(receipts, batches, per_site=False)
    assert all(len(forward) == 1 for forward in made[0].forwards)


def test_failed_wave_cancels_concurrent_waves_and_fails_the_extractor(engine, tmp_path, monkeypatch):
    import nemotron_extraction.vllm as module

    # Request IDs are '<namespace>-<wave>-<row>'; fail the second wave's first request.
    monkeypatch.setattr(module, "uuid", SimpleNamespace(uuid4=lambda: SimpleNamespace(hex="ns")))
    made = engine(budget=3, max_seqs=4, fail="ns-1-0")
    with pytest.raises(RuntimeError, match="engine failure"):
        extract(
            tmp_path,
            waves(3),
            sequences_per_call=2,
            enable_chunked_prefill=True,
            max_num_batched_tokens=3,
            concurrent_waves=True,
        )
    assert not made[0].queue  # the other waves' requests were aborted
    assert not list((tmp_path / "out").glob("*.partial"))


def test_budget_below_model_length_requires_chunked_prefill(tmp_path):
    with pytest.raises(ValueError, match="enable_chunked_prefill"):
        VLLMExtractor("fixture", [HookSite(0, "resid_post")], max_model_len=8, max_num_batched_tokens=4)
