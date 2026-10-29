"""CPU worker RPCs reproduce memory pressure with queued archive waves."""

import asyncio
import sys
from types import SimpleNamespace

import pytest
import torch

from nemotron_extraction import HookSite, VLLMExtractor


@pytest.fixture
def engine(monkeypatch):
    instances = []

    class Engine:
        def __init__(self, args):
            self.args = vars(args)
            self.ready = []
            self.executed = []
            self.windows = 0
            self.live = 900
            self.driver = 980
            self.after = 800
            instances.append(self)

        @classmethod
        def from_engine_args(cls, args):
            return cls(args)

        async def generate(self, *args):
            yield None

        async def abort(self, *args):
            pass

        async def collective_rpc(self, method, args=()):
            if method == "nh_install":
                return [{}]
            if method == "nh_memory_window":
                self.windows += 1
                return [dict(external_device_memory_bytes=0, peak_device_memory_bytes=self.after)]
            if method == "nh_begin":
                self.wave, _, self.tokens, self.path = args
                return [True]
            if method == "nh_seal":
                self.executed.append(len(self.tokens))
                self.ready.append(
                    dict(
                        wave_id=self.wave,
                        path=self.path,
                        token_rows=self.tokens,
                        topology={},
                        peak_gpu_memory_bytes=self.live,
                        peak_gpu_reserved_bytes=self.driver,
                        peak_device_memory_bytes=self.driver,
                        total_device_memory_bytes=1000,
                        peak_device_memory_fraction=self.driver / 1000,
                        actual_forward_batches=[len(self.tokens)],
                        worker_copy_seconds=0,
                        write_seconds=0,
                        validation_seconds=0,
                        serialization_seconds=0,
                        worker_forward_seconds=0,
                    )
                )
                return [True]
            if method == "nh_poll":
                ready, self.ready = self.ready, []
                return [ready]
            return [True]

        def shutdown(self):
            pass

    monkeypatch.setitem(
        sys.modules,
        "vllm",
        SimpleNamespace(AsyncEngineArgs=lambda **k: SimpleNamespace(**k), SamplingParams=lambda **k: k),
    )
    monkeypatch.setitem(sys.modules, "vllm.v1.engine.async_llm", SimpleNamespace(AsyncLLM=Engine))
    return instances


def make(tmp_path, n):
    return VLLMExtractor(
        "fixture",
        [HookSite(0, "resid_post")],
        sequences_per_call=n,
        auto_batch_config=SimpleNamespace(frame_sequences=4),
        sink_dir=tmp_path,
    )


def test_pressure_resizes_once_after_all_eight_waves_and_preserves_order(engine, tmp_path):
    async def run(ex):
        batches = [torch.arange(i * 64, (i + 1) * 64).reshape(32, 2) for i in range(8)]
        first = [r async for r in ex.stream_to_disk(batches, tmp_path / "first")]
        assert len(first) == 8 and [r["index"] for r in first] == list(range(8))
        assert torch.equal(torch.tensor([r["token_rows"] for r in first]), torch.stack(batches))
        assert engine[0].executed == [32] * 8
        assert ex.sequences_per_call == 16
        second = [r async for r in ex.stream_to_disk([torch.ones(16, 2, dtype=torch.long)] * 8, tmp_path / "second")]
        assert len(second) == 8 and engine[0].executed[8:] == [16] * 8
        assert ex.sequences_per_call == 8
        # A fresh phase below the target must not inherit the old memory peak.
        engine[0].driver = engine[0].live = 700
        rows = [r async for r in ex.stream_to_disk([torch.ones(8, 2, dtype=torch.long)], tmp_path / "third")]
        assert len(rows) == 1 and ex.sequences_per_call == 8
        assert ex.last_memory_window["driver_peak_bytes"] == 700

    with make(tmp_path, 32) as ex:
        asyncio.run(run(ex))


@pytest.mark.parametrize("live,raises", [(900, False), (970, True)])
def test_minimum_frame_cache_pressure_vs_actual_live_exhaustion(engine, tmp_path, live, raises):
    async def run(ex):
        seen = []

        async def consume():
            async for r in ex.stream_to_disk([torch.ones(4, 2, dtype=torch.long)] * 8, tmp_path / "out"):
                seen.append(r)

        engine[0].live = live
        if raises:
            with pytest.raises(MemoryError, match="archive frame"):
                await consume()
        else:
            await consume()
        assert len(seen) == 8  # Durable publications survive the final capacity error.
        assert engine[0].executed == [4] * 8

    with make(tmp_path, 4) as ex:
        asyncio.run(run(ex))


def test_explicit_setter_preserves_engine_capacity(engine, tmp_path):
    with make(tmp_path, 16) as ex:
        ex.set_sequences_per_call(4)
        assert ex.sequences_per_call == 4 and engine[0].args["max_num_seqs"] == 16
        for n in [0, 2, 20, True]:
            with pytest.raises(ValueError):
                ex.set_sequences_per_call(n)
