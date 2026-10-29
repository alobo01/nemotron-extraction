from __future__ import annotations

import asyncio
import tempfile
import threading
import time
import uuid
from dataclasses import asdict

import torch
from safetensors.torch import load_file

from .extractor import Extractor
from .precision import checkpoint_directory, validate_precision
from .transport import SLOTS
from .types import ActivationBatch, validate_tokens

# Explicit Mamba cache types: the SSM state stays float32, the convolution state bfloat16.
MATCHED_CACHE_DTYPES = dict(mamba_ssm_cache_dtype="float32", mamba_cache_dtype="bfloat16")


class VLLMExtractor(Extractor):
    """Persistent AsyncLLM workers with bounded in-flight capture/publication slots."""

    _worker_extension_cls = "nemotron_extraction.worker.ActivationHookExtension"

    def __init__(
        self,
        model,
        sites,
        *,
        revision=None,
        max_model_len=2049,
        sequences_per_call=4,
        dtype="bfloat16",
        trust_remote_code=True,
        enable_chunked_prefill=False,
        sink_dir=None,
        gpu_memory_utilization=0.96,
        kv_cache_memory_bytes=None,
        diagnostics=False,
        publication_threads=1,
        auto_batch_config=None,
        publication_layout="monolithic",
        site_write_threads=8,
        max_num_batched_tokens=None,
        concurrent_waves=False,
        capture_slots=SLOTS,
        precision="native",
    ):
        """Waves are serialised unless ``concurrent_waves``; then up to ``capture_slots`` waves hold
        buffers at once and the scheduler may mix their prompts in one forward. A token budget
        below ``max_model_len`` needs ``enable_chunked_prefill``; prompts may then span forwards.
        """
        super().__init__(sites, diagnostics, publication_threads, publication_layout, site_write_threads)
        self.precision = validate_precision(precision)
        if type(capture_slots) is not int or capture_slots < 2:
            raise ValueError("capture_slots must be an integer of at least 2")
        if max_num_batched_tokens is not None and max_num_batched_tokens < 1:
            raise ValueError("max_num_batched_tokens must be positive")
        if max_num_batched_tokens is not None and max_num_batched_tokens < max_model_len and not enable_chunked_prefill:
            raise ValueError("A token budget below max_model_len requires enable_chunked_prefill")
        self.max_num_batched_tokens = max_num_batched_tokens
        self.concurrent_waves, self.capture_slots = concurrent_waves, capture_slots
        self.precision_receipt = None
        automatic = sequences_per_call == "auto"
        if (not automatic and sequences_per_call < 1) or max_model_len < 2 or not 0 < gpu_memory_utilization <= 1:
            raise ValueError("Invalid sequence batch size, model length or memory utilization")
        self.auto_batch_config = auto_batch_config
        if automatic:
            if auto_batch_config is None:
                raise ValueError("Automatic extraction batching needs an AutoBatchConfig memory profile")
            free, total = torch.cuda.mem_get_info()
            sequences_per_call = auto_batch_config.fit(total, gpu_memory_utilization, external_used_bytes=total - free)
        self.last_memory_window = {}
        self.sequence_ceiling = sequences_per_call
        self.sequences_per_call, self.max_model_len = sequences_per_call, max_model_len
        self.enable_chunked_prefill = enable_chunked_prefill
        self.llm, self.installed = None, False
        self.gpu_memory_utilization = gpu_memory_utilization
        self.kv_cache_memory_bytes = kv_cache_memory_bytes
        self.engine_startup_seconds = self.hook_install_seconds = 0.0
        self.generate_seconds = self.worker_forward_seconds = self.worker_collection_seconds = 0.0
        self.worker_copy_seconds = self.worker_rpc_seconds = 0.0
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

    def set_active_sites(self, sites):
        """Reduce captured publications at a drained boundary without reloading the model."""
        keys = [site.key for site in sites]

        async def update():
            async with self.lock:
                await self.llm.collective_rpc("nh_set_sites", args=(keys,))

        return self._future(update()).result()

    def set_sequences_per_call(self, sequences):
        """Set the next call's geometry after all current publications have drained.

        The engine capacity stays fixed, so changing extraction wave size never
        reloads the model. Archive users must build reservations after this call.
        """
        frame = self.auto_batch_config.frame_sequences if self.auto_batch_config else 1
        if type(sequences) is not int or not frame <= sequences <= self.sequence_ceiling or sequences % frame:
            raise ValueError("Batch must be frame-aligned within the initialized engine capacity")

        async def update():
            async with self.lock:
                self._require_open()
                if self.failed:
                    raise RuntimeError("Cannot resize a failed extractor")
                self.sequences_per_call = sequences

        return self._future(update()).result()

    def _future(self, coroutine):
        return asyncio.run_coroutine_threadsafe(coroutine, self.loop)

    async def _start(self, model, revision, dtype, trust_remote_code):
        from vllm import AsyncEngineArgs, SamplingParams
        from vllm.v1.engine.async_llm import AsyncLLM

        self.lock = asyncio.Lock()
        started = time.perf_counter()
        args = AsyncEngineArgs(
            model=model,
            revision=revision,
            tokenizer_revision=revision,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            tensor_parallel_size=1,
            enforce_eager=True,
            enable_prefix_caching=False,
            enable_chunked_prefill=self.enable_chunked_prefill,
            skip_tokenizer_init=True,
            max_model_len=self.max_model_len,
            max_num_seqs=self.sequences_per_call,
            max_num_batched_tokens=self.max_num_batched_tokens or self.sequences_per_call * self.max_model_len,
            gpu_memory_utilization=self.gpu_memory_utilization,
            kv_cache_memory_bytes=self.kv_cache_memory_bytes,
            worker_extension_cls=self._worker_extension_cls,
            **(MATCHED_CACHE_DTYPES if self.precision == "matched" else {}),
        )
        self.llm = AsyncLLM.from_engine_args(args)
        self.engine_startup_seconds = time.perf_counter() - started
        self.params = SamplingParams(max_tokens=1, temperature=0)
        started = time.perf_counter()
        self.topology = (
            await self.llm.collective_rpc(
                "nh_install",
                args=(
                    [asdict(s) for s in self.sites],
                    self.sink.name,
                    self.diagnostics,
                    self.publication_threads,
                    self.publication_layout,
                    self.site_write_threads,
                    self.capture_slots,
                ),
            )
        )[0]
        if self.precision == "matched":
            checkpoint = str(checkpoint_directory(model, revision))
            self.precision_receipt = (await self.llm.collective_rpc("nh_match_precision", args=(checkpoint,)))[0]
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
            self._fold_memory(receipt)
            if self.auto_batch_config is not None:
                self._phase_total = max(self._phase_total, receipt.get("total_device_memory_bytes", 0))
                self._phase_driver = max(self._phase_driver, receipt.get("peak_device_memory_bytes", 0))
                self._phase_reserved = max(self._phase_reserved, receipt.get("peak_gpu_reserved_bytes", 0))
                self._phase_live = max(
                    self._phase_live,
                    receipt.get("peak_gpu_memory_bytes", 0) + self._phase_external,
                )
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
            call_sequences = self.sequences_per_call
            self._phase_total = self._phase_driver = self._phase_reserved = self._phase_live = 0
            self._phase_external = 0
            if self.auto_batch_config is not None:
                baseline = (await self.llm.collective_rpc("nh_memory_window"))[0]
                self._phase_external = baseline["external_device_memory_bytes"]
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

            running = set()
            generating = dict(waves=0, since=0.0)

            def reap():
                """Remove finished capture tasks and raise the first failure."""
                for task in [task for task in running if task.done()]:
                    running.discard(task)
                    task.result()

            async def drain(limit):
                """Poll until at most `limit` waves hold buffers; return seconds waited."""
                started = time.perf_counter()
                while len(pending) > limit:
                    reap()
                    await collect()
                    if len(pending) > limit:
                        await asyncio.sleep(0.02)
                return time.perf_counter() - started

            async def capture(wave_id, chunk, ids):
                # Generation time is wall time with at least one wave in flight.
                if not generating["waves"]:
                    generating["since"] = time.perf_counter()
                generating["waves"] += 1
                try:
                    await self._generate(chunk, ids)
                finally:
                    generating["waves"] -= 1
                    if not generating["waves"]:
                        self.generate_seconds += time.perf_counter() - generating["since"]
                rpc_started = time.perf_counter()
                await self.llm.collective_rpc("nh_seal", args=(wave_id,))
                self.worker_rpc_seconds += time.perf_counter() - rpc_started

            try:
                index = 0
                for batch in token_batches:
                    tokens = validate_tokens(batch)
                    if tokens.shape[1] + 1 > self.max_model_len:
                        raise ValueError("Prompt plus one generated token exceeds max_model_len")
                    for chunk in tokens.split(call_sequences):
                        if len(pending) >= self.capture_slots:
                            self.backpressure_seconds += await drain(self.capture_slots - 1)
                        path = self.shard_path(output_dir, index)
                        if path.exists():
                            raise FileExistsError(path)
                        wave_id = f"{namespace}-{index}"
                        ids = [f"{wave_id}-{row}" for row in range(len(chunk))]
                        rpc_started = time.perf_counter()
                        admitted = (
                            await self.llm.collective_rpc(
                                "nh_begin", args=(wave_id, ids, chunk.tolist(), str(path.resolve()))
                            )
                        )[0]
                        self.worker_rpc_seconds += time.perf_counter() - rpc_started
                        if not admitted:
                            raise RuntimeError("Capture admission violated the buffer-slot bound")
                        pending[wave_id] = index
                        if self.concurrent_waves:
                            running.add(asyncio.create_task(capture(wave_id, chunk, ids)))
                        else:
                            await capture(wave_id, chunk, ids)
                        index += 1
                while running:
                    await asyncio.wait(running, return_when=asyncio.FIRST_COMPLETED)
                    reap()
            except BaseException:
                for task in running:
                    task.cancel()
                await asyncio.gather(*running, return_exceptions=True)
                raise
            self.final_drain_seconds += await drain(0)
            if self.auto_batch_config is not None:
                target = self.gpu_memory_utilization * self._phase_total
                pressure = bool(self._phase_total and max(self._phase_driver, self._phase_live) > target)
                current_used = 0
                if pressure:
                    # Only unused allocator blocks are released; models stay resident.
                    after = (await self.llm.collective_rpc("nh_memory_window", args=(True,)))[0]
                    current_used = after["peak_device_memory_bytes"]
                    frame = self.auto_batch_config.frame_sequences
                    if call_sequences <= frame:
                        if self._phase_live > target or current_used > target:
                            raise MemoryError("One extraction archive frame exceeds the device memory target")
                    else:
                        # One reduction per drained call, based on the geometry
                        # actually executed, never once per queued receipt.
                        self.sequences_per_call = max(frame, call_sequences // 2 // frame * frame)
                self.last_memory_window = dict(
                    sequences=call_sequences,
                    next_sequences=self.sequences_per_call,
                    live_peak_bytes=self._phase_live,
                    reserved_peak_bytes=self._phase_reserved,
                    driver_peak_bytes=self._phase_driver,
                    external_bytes=self._phase_external,
                    total_bytes=self._phase_total,
                    driver_after_trim_bytes=current_used,
                    pressure=pressure,
                )

    async def stream_to_disk(self, token_batches, output_dir):
        """Yield ordered durable shard receipts; two publications may overlap inference."""
        caller = asyncio.get_running_loop()
        messages = asyncio.Queue(maxsize=SLOTS)
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
                if receipt.get("layout") == "per_site":
                    tensors = {}
                    for key, record in receipt["files"].items():
                        tensors.update(await asyncio.to_thread(load_file, record["path"]))
                else:
                    tensors = await asyncio.to_thread(load_file, receipt["path"])
                for key in parts:
                    parts[key].append(tensors[key])
            self.driver_read_seconds += time.perf_counter() - started
            return ActivationBatch(tokens, {key: torch.cat(values) for key, values in parts.items()})

    def _submit_extract(self, token_ids):
        self._require_open()
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
