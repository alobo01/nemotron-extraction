import asyncio
import contextlib
import threading

import pytest
import torch
from safetensors.torch import load_file
from tiny_models import Model

from nemotron_extraction import HFExtractor, HookSite
from nemotron_extraction.transport import BufferPool, publish


def test_two_slots_are_reused_only_after_release():
    pool = BufferPool()
    a, b = pool.acquire(), pool.acquire()
    assert pool.acquire() is None
    tensor = pool.tensor(a, "x", (2, 3), torch.float32, pinned=False)
    pool.release(a)
    assert pool.acquire() == a
    assert pool.tensor(a, "x", (2, 3), torch.float32, pinned=False) is tensor
    assert pool.tensor(a, "x", (1, 3), torch.float32, pinned=False).shape == (1, 3)
    pool.release(b)
    with pytest.raises(RuntimeError, match="twice"):
        pool.release(b)


@pytest.mark.parametrize("publication_threads", [1, 2])
def test_hf_async_extract_and_direct_disk_rows(tmp_path, publication_threads):
    async def run(extractor):
        tokens = torch.tensor([[1, 2], [1, 2], [4, 5]])
        batch = await extractor.extract_async(tokens)
        assert torch.equal(batch.token_ids, tokens)
        batches = [tokens, tokens + 10, torch.tensor([[7]])]
        receipts = [item async for item in extractor.stream_to_disk(batches, tmp_path)]
        assert [item["index"] for item in receipts] == [0, 1, 2]
        for expected, receipt in zip(batches, receipts):
            saved = load_file(receipt["path"])
            assert torch.equal(saved["token_ids"], expected)
            assert saved["L0_resid_post"].shape == (expected.numel(), 2)
        assert torch.equal(batch.token_ids, tokens)

    with HFExtractor(
        Model(), [HookSite(0, "resid_post")], device="cpu", publication_threads=publication_threads
    ) as extractor:
        asyncio.run(run(extractor))


@pytest.mark.parametrize("publication_threads", [1, 2])
def test_hf_writer_failure_releases_buffers(monkeypatch, tmp_path, publication_threads):
    import nemotron_extraction.hf as hf

    monkeypatch.setattr(hf, "publish", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))

    async def run(extractor):
        with pytest.raises(OSError, match="disk full"):
            async for _ in extractor.stream_to_disk([torch.tensor([[1, 2]])], tmp_path):
                pass

    with HFExtractor(
        Model(), [HookSite(0, "resid_post")], device="cpu", publication_threads=publication_threads
    ) as extractor:
        asyncio.run(run(extractor))
        assert sorted(extractor.pool.free) == [0, 1]


def test_inference_advances_while_writer_is_blocked(monkeypatch, tmp_path):
    import nemotron_extraction.hf as hf

    entered, next_forward, release = threading.Event(), threading.Event(), threading.Event()
    real_publish = hf.publish

    def blocked(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return real_publish(*args, **kwargs)

    monkeypatch.setattr(hf, "publish", blocked)
    model = Model()
    calls = []

    def observed(module, args, output):
        calls.append(1)
        if len(calls) == 2:
            next_forward.set()

    handle = model.model.register_forward_hook(observed)

    async def run(extractor):
        async def consume():
            return [r async for r in extractor.stream_to_disk([torch.tensor([[1]]), torch.tensor([[2]])], tmp_path)]

        task = asyncio.create_task(consume())
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            assert await asyncio.to_thread(next_forward.wait, 5)
        finally:
            release.set()
        assert len(await task) == 2

    try:
        with HFExtractor(model, [HookSite(0, "resid_post")], device="cpu") as extractor:
            asyncio.run(run(extractor))
    finally:
        handle.remove()


def test_atomic_failure_leaves_no_published_shard(tmp_path):
    path = tmp_path / "bad.safetensors"
    with pytest.raises(ValueError, match="nonfinite"):
        publish({"x": torch.tensor([float("nan")])}, path)
    assert not path.exists() and not path.with_suffix(".safetensors.partial").exists()


class _EngineLoopExtractor:
    """A VLLMExtractor shell with a real background loop and lock but a scripted `_stream`."""

    def __init__(self, script):
        from nemotron_extraction import VLLMExtractor

        self.extractor = VLLMExtractor.__new__(VLLMExtractor)
        self.extractor.loop = loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=loop.run_forever, daemon=True)
        self.thread.start()

        async def make_lock():
            return asyncio.Lock()

        self.extractor.lock = asyncio.run_coroutine_threadsafe(make_lock(), loop).result(5)

        async def stream(token_batches, output_dir, emit):
            async with self.extractor.lock:
                await script(emit)

        self.extractor._stream = stream

    def __enter__(self):
        return self.extractor

    def __exit__(self, *exc):
        self.extractor.loop.call_soon_threadsafe(self.extractor.loop.stop)
        self.thread.join(5)
        self.extractor.loop.close()


def test_vllm_stream_orders_receipts_then_raises_after_draining():
    async def script(emit):
        for index in range(5):
            await emit({"index": index})
        raise ValueError("boom")

    async def run(extractor):
        seen = []
        with pytest.raises(ValueError, match="boom"):
            async for receipt in extractor.stream_to_disk([], None):
                seen.append(receipt["index"])
                await asyncio.sleep(0)
        assert seen == [0, 1, 2, 3, 4]

    with _EngineLoopExtractor(script) as extractor:
        asyncio.run(run(extractor))


def test_vllm_stream_buffers_at_most_two_receipts():
    entered, reached = [], threading.Event()

    async def script(emit):
        for index in range(6):
            entered.append(index)
            if index == 3:
                reached.set()
            await emit({"index": index})

    async def run(extractor):
        stream = extractor.stream_to_disk([], None)
        assert (await anext(stream))["index"] == 0
        assert await asyncio.to_thread(reached.wait, 5)
        await asyncio.sleep(0.1)
        assert entered == [0, 1, 2, 3]  # receipt 3 waits for room; 4 is never reached
        assert [r["index"] async for r in stream] == [1, 2, 3, 4, 5]

    with _EngineLoopExtractor(script) as extractor:
        asyncio.run(run(extractor))


def test_vllm_consumer_exit_releases_producer_blocked_on_full_buffer():
    blocked, cancelled, finished = threading.Event(), threading.Event(), threading.Event()

    async def script(emit):
        try:
            for index in range(10):
                if index == 3:
                    blocked.set()
                await emit({"index": index})
        except asyncio.CancelledError:
            cancelled.set()
            raise
        finally:
            finished.set()

    async def run(extractor):
        stream = extractor.stream_to_disk([], None)
        assert (await anext(stream))["index"] == 0
        assert await asyncio.to_thread(blocked.wait, 5)
        await asyncio.sleep(0.05)
        await stream.aclose()
        assert cancelled.is_set() and finished.is_set()
        assert not extractor.lock.locked()
        await asyncio.sleep(0.05)
        assert asyncio.all_tasks() == {asyncio.current_task()}  # no cross-loop put survives

    with _EngineLoopExtractor(script) as extractor:
        asyncio.run(run(extractor))


def test_vllm_consumer_cancellation_releases_blocked_producer():
    blocked, cancelled = threading.Event(), threading.Event()

    async def script(emit):
        try:
            for index in range(10):
                if index == 3:
                    blocked.set()
                await emit({"index": index})
        except asyncio.CancelledError:
            cancelled.set()
            raise

    async def run(extractor):
        async def consume():
            async with contextlib.aclosing(extractor.stream_to_disk([], None)) as stream:
                async for _ in stream:
                    await asyncio.Event().wait()  # never asks for the next receipt

        task = asyncio.create_task(consume())
        assert await asyncio.to_thread(blocked.wait, 5)
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert cancelled.is_set() and not extractor.lock.locked()
        await asyncio.sleep(0.05)
        assert asyncio.all_tasks() == {asyncio.current_task()}

    with _EngineLoopExtractor(script) as extractor:
        asyncio.run(run(extractor))
