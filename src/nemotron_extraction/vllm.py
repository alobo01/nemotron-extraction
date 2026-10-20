from __future__ import annotations
import asyncio
import tempfile
import threading
import time
import uuid
from dataclasses import asdict
from pathlib import Path
import torch
from safetensors.torch import load_file
from .transport import MEMORY_KEYS
from .types import ActivationBatch, validate_publication_threads, validate_sites, validate_tokens


class VLLMExtractor:
    """Persistent AsyncLLM workers with two in-flight capture/publication slots."""
    _worker_extension_cls = "nemotron_extraction.worker.ActivationHookExtension"

    def __init__(self, model, sites, *, revision=None, max_model_len=2049,
                 sequences_per_call=4, dtype="bfloat16", trust_remote_code=True,
                 enable_chunked_prefill=False, sink_dir=None, gpu_memory_utilization=0.96,
                 kv_cache_memory_bytes=None, diagnostics=False, publication_threads=1):
        self.sites = validate_sites(sites)
        self.diagnostics = diagnostics
        self.publication_threads = validate_publication_threads(publication_threads)
        if sequences_per_call < 1 or max_model_len < 2 or not 0 < gpu_memory_utilization <= 1:
            raise ValueError("Invalid sequence batch size, model length or memory utilization")
        self.sequences_per_call, self.max_model_len = sequences_per_call, max_model_len
        self.enable_chunked_prefill = enable_chunked_prefill
        self.closed, self.llm, self.installed = False, None, False
        self.gpu_memory_utilization = gpu_memory_utilization
        self.kv_cache_memory_bytes = kv_cache_memory_bytes
        self.engine_startup_seconds = self.hook_install_seconds = 0.0
        self.generate_seconds = self.worker_forward_seconds = 0.0
        self.worker_collection_seconds = self.driver_read_seconds = 0.0
        self.worker_copy_seconds = self.worker_write_seconds = 0.0
        self.validation_seconds = self.serialization_seconds = self.worker_rpc_seconds = 0.0
        self.backpressure_seconds = self.final_drain_seconds = 0.0
        self.peak_gpu_memory_bytes = self.peak_gpu_reserved_bytes = self.peak_device_memory_bytes = 0
        self.total_device_memory_bytes = 0
        self.peak_device_memory_fraction = 0.0
        self.actual_forward_batches = []
        self.failed = False
        self.tasks = set()
        self.sink = tempfile.TemporaryDirectory(prefix="nemotron-hooks-", dir=sink_dir)
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()
        try:
            self._future(self._start(model, revision, dtype, trust_remote_code)).result()
        except BaseException:
            self.close()
            raise

    def _future(self, coroutine):
        return asyncio.run_coroutine_threadsafe(coroutine, self.loop)

    async def _start(self, model, revision, dtype, trust_remote_code):
        from vllm import AsyncEngineArgs, SamplingParams
        from vllm.v1.engine.async_llm import AsyncLLM
        self.lock = asyncio.Lock()
        started = time.perf_counter()
        args = AsyncEngineArgs(
            model=model, revision=revision, tokenizer_revision=revision, dtype=dtype,
            trust_remote_code=trust_remote_code, tensor_parallel_size=1, enforce_eager=True,
            enable_prefix_caching=False, enable_chunked_prefill=self.enable_chunked_prefill,
            skip_tokenizer_init=True,
            max_model_len=self.max_model_len, max_num_seqs=self.sequences_per_call,
            max_num_batched_tokens=self.sequences_per_call * self.max_model_len,
            gpu_memory_utilization=self.gpu_memory_utilization,
            kv_cache_memory_bytes=self.kv_cache_memory_bytes,
            worker_extension_cls=self._worker_extension_cls)
        self.llm = AsyncLLM.from_engine_args(args)
        self.engine_startup_seconds = time.perf_counter() - started
        self.params = SamplingParams(max_tokens=1, temperature=0)
        started = time.perf_counter()
        self.topology = (await self.llm.collective_rpc(
            "nh_install", args=([asdict(s) for s in self.sites], self.sink.name,
                                self.diagnostics, self.publication_threads)))[0]
        self.hook_install_seconds = time.perf_counter() - started
        self.installed = True

    async def _poll(self):
        started = time.perf_counter()
        ready = (await self.llm.collective_rpc("nh_poll"))[0]
        self.worker_collection_seconds += time.perf_counter() - started
        self.worker_rpc_seconds += time.perf_counter() - started
        for receipt in ready:
            self.topology = receipt["topology"]
            self.worker_copy_seconds += receipt["worker_copy_seconds"]
            self.worker_write_seconds += receipt["write_seconds"]
            self.validation_seconds += receipt["validation_seconds"]
            self.serialization_seconds += receipt["serialization_seconds"]
            self.worker_forward_seconds += receipt["worker_forward_seconds"]
            self.actual_forward_batches.extend(receipt["actual_forward_batches"])
            for name in MEMORY_KEYS:
                setattr(self, name, max(getattr(self, name), receipt[name]))
        return ready

    async def _generate(self, tokens, ids):
        async def request(row, rid):
            async for _ in self.llm.generate({"prompt_token_ids": row}, self.params, rid):
                pass
        tasks = [asyncio.create_task(request(row, rid)) for row, rid in zip(tokens.tolist(), ids)]
        try:
            await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await asyncio.gather(*(self.llm.abort(rid) for rid in ids), return_exceptions=True)
            raise

    async def _stream(self, token_batches, output_dir, emit):
        if self.closed or self.failed:
            raise RuntimeError("Extractor is closed or failed; create a new extractor")
        task = asyncio.current_task()
        self.tasks.add(task)
        try:
            await self._stream_inner(token_batches, output_dir, emit)
        except BaseException:
            self.failed = True
            raise
        finally:
            self.tasks.discard(task)

    async def _stream_inner(self, token_batches, output_dir, emit):
        async with self.lock:
            pending, completed = {}, {}
            next_yield = 0
            namespace = uuid.uuid4().hex
            async def collect():
                nonlocal next_yield
                for receipt in await self._poll():
                    index = pending.pop(receipt["wave_id"])
                    receipt["index"] = index
                    completed[index] = receipt
                while next_yield in completed:
                    await emit(completed.pop(next_yield))
                    next_yield += 1

            async def drain(limit):
                """Poll until at most `limit` publications are pending; return seconds waited."""
                started = time.perf_counter()
                while len(pending) > limit:
                    await collect()
                    if len(pending) > limit:
                        await asyncio.sleep(.02)
                return time.perf_counter() - started

            index = 0
            for batch in token_batches:
                tokens = validate_tokens(batch)
                if tokens.shape[1] + 1 > self.max_model_len:
                    raise ValueError("Prompt plus one generated token exceeds max_model_len")
                for chunk in tokens.split(self.sequences_per_call):
                    if len(pending) >= 2:
                        self.backpressure_seconds += await drain(1)
                    path = Path(output_dir) / f"{index:06d}.safetensors"
                    if path.exists():
                        raise FileExistsError(path)
                    wave_id = f"{namespace}-{index}"
                    ids = [f"{wave_id}-{row}" for row in range(len(chunk))]
                    rpc_started = time.perf_counter()
                    admitted = (await self.llm.collective_rpc(
                        "nh_begin", args=(wave_id, ids, chunk.tolist(), str(path.resolve()))))[0]
                    self.worker_rpc_seconds += time.perf_counter() - rpc_started
                    if not admitted:
                        raise RuntimeError("Capture admission violated the two-slot bound")
                    pending[wave_id] = index
                    started = time.perf_counter()
                    await self._generate(chunk, ids)
                    self.generate_seconds += time.perf_counter() - started
                    rpc_started = time.perf_counter()
                    await self.llm.collective_rpc("nh_seal", args=(wave_id,))
                    self.worker_rpc_seconds += time.perf_counter() - rpc_started
                    index += 1
            self.final_drain_seconds += await drain(0)

    async def stream_to_disk(self, token_batches, output_dir):
        """Yield ordered durable shard receipts; two publications may overlap inference."""
        caller = asyncio.get_running_loop()
        messages = asyncio.Queue(maxsize=2)
        sending = []
        async def emit(receipt):
            # Runs on the engine loop; the put runs on the caller's loop and blocks there while two
            # receipts wait. Cancelling this await cancels that put through the wrapped future.
            put = asyncio.run_coroutine_threadsafe(messages.put(receipt), caller)
            sending[:] = [put]
            await asyncio.wrap_future(put)
        def abandon_put():
            for put in list(sending):
                put.cancel()
        future = self._future(self._stream(token_batches, output_dir, emit))
        finished = asyncio.wrap_future(future)
        getter = None
        try:
            while True:
                try:
                    receipt = messages.get_nowait()
                except asyncio.QueueEmpty:
                    if finished.done():
                        break
                    getter = asyncio.ensure_future(messages.get())
                    await asyncio.wait({getter, finished}, return_when=asyncio.FIRST_COMPLETED)
                    if not getter.done():
                        getter.cancel()
                        await asyncio.gather(getter, return_exceptions=True)
                    if getter.cancelled():
                        getter = None
                        continue
                    receipt, getter = getter.result(), None
                if finished.done():
                    abandon_put()  # a put cut short by cancellation must not slip in later
                yield receipt
            abandon_put()
            await finished
        finally:
            if getter is not None:
                getter.cancel()
            if not future.done():
                future.cancel()
                # The stream owns the worker until canceled tasks have unwound.
                async def barrier():
                    async with self.lock:
                        pass
                await asyncio.wrap_future(self._future(barrier()))
            abandon_put()
            if finished.done() and not finished.cancelled():
                finished.exception()  # mark retrieved when the consumer left before awaiting it

    async def _extract(self, token_ids):
        tokens = validate_tokens(token_ids)
        with tempfile.TemporaryDirectory(dir=self.sink.name) as directory:
            receipts = []
            async def emit(receipt):
                receipts.append(receipt)
            await self._stream([tokens], directory, emit)
            started = time.perf_counter()
            parts = {site.key: [] for site in self.sites}
            for receipt in receipts:
                tensors = await asyncio.to_thread(load_file, receipt["path"])
                for key in parts:
                    parts[key].append(tensors[key])
            self.driver_read_seconds += time.perf_counter() - started
            return ActivationBatch(tokens, {key: torch.cat(values) for key, values in parts.items()})

    def _submit_extract(self, token_ids):
        if self.closed:
            raise RuntimeError("Extractor is closed")
        return self._future(self._extract(token_ids))

    def extract(self, token_ids):
        return self._submit_extract(token_ids).result()

    async def extract_async(self, token_ids):
        return await asyncio.wrap_future(self._submit_extract(token_ids))

    async def _close(self):
        for task in list(self.tasks):
            task.cancel()
        if self.tasks:
            await asyncio.gather(*list(self.tasks), return_exceptions=True)
        if self.llm is not None:
            try:
                if self.installed:
                    await self.llm.collective_rpc("nh_remove")
            finally:
                self.llm.shutdown()
                self.llm = None

    def close(self):
        if self.closed:
            return
        try:
            self._future(self._close()).result()
        finally:
            self.closed = True
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.thread.join()
            self.loop.close()
            self.sink.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
