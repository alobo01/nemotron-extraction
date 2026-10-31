"""Per-group token sums of block outputs, reduced on the GPU inside the vLLM workers.

Activations never leave the device: every forward adds the scheduled prompt tokens of each group
to a float64 accumulator of shape [layers, groups, width]. This replaces capture, copy and
publication when only a mean per group is needed, as for language vectors.
"""

from __future__ import annotations

import asyncio
import time
from functools import partial

import torch

from .hooks import layers_of
from .worker import external_request_id


def group_weights(spans, rows, groups):
    """Float32 [groups, rows] indicator of which group each flat-batch row belongs to.

    ``spans`` holds (start, count, group); rows outside every span, such as padding, belong to no group.
    """
    index = torch.full((rows,), groups, dtype=torch.long)
    for start, count, group in spans:
        index[start : start + count] = group
    return (index[None, :] == torch.arange(groups)[:, None]).float()


def decode_result(payload):
    """Sums [layers, groups, width] in float64 and counts [groups] from `ag_result`."""
    data, shape, counts = payload
    return torch.frombuffer(bytearray(data), dtype=torch.float64).reshape(shape), torch.tensor(counts)


class AggregateHookExtension:
    """vLLM worker extension that sums block outputs per group of requests."""

    def ag_install(self, layers, groups):
        _, blocks = layers_of(self.model_runner.model)
        if max(layers) >= len(blocks):
            raise ValueError(f"Model has {len(blocks)} layers, not {max(layers) + 1}")
        self.ag_layers, self.ag_groups = list(layers), groups
        self.ag_sums = None
        self.ag_counts = torch.zeros(groups, dtype=torch.long)
        self.ag_requests, self.ag_spans, self.ag_weights = {}, [], None
        model = self.model_runner.model
        self.ag_handles = [
            model.register_forward_pre_hook(self._ag_inputs, with_kwargs=True),
            model.register_forward_hook(self._ag_end),
        ]
        for slot, layer in enumerate(self.ag_layers):
            hook = partial(self._ag_block, slot)
            self.ag_handles.append(blocks[layer].register_forward_hook(hook))
        return len(blocks)

    def ag_begin(self, requests):
        """Register {request ID: (group, prompt length)} before the requests are submitted."""
        if any(rid in self.ag_requests for rid in requests):
            raise ValueError("Request ID already registered")
        self.ag_requests.update((rid, [group, length, 0]) for rid, (group, length) in requests.items())
        return len(requests)

    def _ag_inputs(self, module, args, kwargs):
        runner = self.model_runner
        batch = runner.input_batch
        ids = list(batch.req_ids)
        starts = runner.query_start_loc.cpu[: len(ids) + 1].tolist()
        spans = []
        for index, rid in enumerate(ids):
            entry = self.ag_requests[external_request_id(rid, self.ag_requests)]
            group, length, cursor = entry
            computed = int(batch.num_computed_tokens_cpu[index])
            if computed >= length:
                continue  # generated tokens are never counted
            if computed != cursor:
                raise RuntimeError(f"Prompt span out of order for {rid}: expected offset {cursor}, got {computed}")
            count = min(starts[index + 1] - starts[index], length - computed)
            spans.append((starts[index], count, group, entry))
        # Commit only after every request in this forward was accepted.
        for _, count, group, entry in spans:
            entry[2] += count
            self.ag_counts[group] += count
        self.ag_spans = [span[:3] for span in spans]
        self.ag_weights = None

    def _ag_block(self, slot, module, args, output):
        if not self.ag_spans:
            return
        hidden, running = output
        value = hidden + running
        if self.ag_weights is None or self.ag_weights.shape[1] != value.shape[0]:
            if max(start + count for start, count, _ in self.ag_spans) > value.shape[0]:
                raise RuntimeError("Block output does not cover scheduler spans")
            self.ag_weights = group_weights(self.ag_spans, value.shape[0], self.ag_groups).to(value.device)
        if self.ag_sums is None:
            shape = (len(self.ag_layers), self.ag_groups, value.shape[-1])
            self.ag_sums = torch.zeros(shape, dtype=torch.float64, device=value.device)
        # Exact float32 products with 0/1 weights; TF32 would round the activations.
        precision = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision("highest")
        try:
            self.ag_sums[slot] += self.ag_weights @ value.float()
        finally:
            torch.set_float32_matmul_precision(precision)

    def _ag_end(self, module, args, output):
        self.ag_spans, self.ag_weights = [], None

    def ag_result(self):
        """Sums and counts as plain bytes and lists, which survive vLLM's RPC serialisation unchanged."""
        if any(cursor != length for _, length, cursor in self.ag_requests.values()):
            raise RuntimeError("Missing prompt activations")
        if self.ag_sums is None:
            raise RuntimeError("No tokens were counted")
        torch.cuda.synchronize()
        sums = self.ag_sums.cpu()
        return sums.numpy().tobytes(), list(sums.shape), self.ag_counts.tolist()

    def ag_reset(self):
        """Forget all counted tokens and registered requests, e.g. after a warm-up pass."""
        if self.ag_sums is not None:
            self.ag_sums.zero_()
        self.ag_counts.zero_()
        self.ag_requests.clear()
        return True

    def ag_remove(self):
        for handle in self.ag_handles:
            handle.remove()
        self.ag_handles.clear()
        self.ag_requests.clear()


class AggregateExtractor:
    """Group means of block outputs over many prompts; use as ``async with`` inside one event loop.

    Chunked prefill and continuous batching apply as in `VLLMExtractor`: prompts of any length
    share forwards up to ``max_num_batched_tokens``.
    """

    def __init__(
        self,
        model,
        layers,
        groups,
        *,
        revision=None,
        dtype="bfloat16",
        tensor_parallel_size=1,
        max_model_len=2048,
        max_num_batched_tokens=16384,
        max_num_seqs=512,
        gpu_memory_utilization=0.9,
        enable_chunked_prefill=True,
        trust_remote_code=True,
        engine_options=None,
    ):
        """``engine_options`` are further `AsyncEngineArgs` fields, for example ``hf_overrides``."""
        if groups < 1 or not layers or len(set(layers)) != len(layers) or min(layers) < 0:
            raise ValueError("Provide at least one group and distinct nonnegative layers")
        self.layers, self.groups = list(layers), groups
        self.model, self.revision, self.dtype = model, revision, dtype
        self.options = dict(
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_num_seqs,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_chunked_prefill=enable_chunked_prefill,
            trust_remote_code=trust_remote_code,
            **(engine_options or {}),
        )
        self.llm, self.installed = None, False
        self.engine_startup_seconds = self.hook_install_seconds = self.warmup_seconds = self.extract_seconds = 0.0

    async def __aenter__(self):
        from vllm import AsyncEngineArgs, SamplingParams
        from vllm.v1.engine.async_llm import AsyncLLM

        started = time.perf_counter()
        args = AsyncEngineArgs(
            model=self.model,
            revision=self.revision,
            tokenizer_revision=self.revision,
            dtype=self.dtype,
            enforce_eager=True,
            enable_prefix_caching=False,
            skip_tokenizer_init=True,
            worker_extension_cls="nemotron_extraction.aggregate.AggregateHookExtension",
            **self.options,
        )
        self.llm = AsyncLLM.from_engine_args(args)
        self.engine_startup_seconds = time.perf_counter() - started
        self.params = SamplingParams(max_tokens=1, temperature=0)
        started = time.perf_counter()
        self.layer_count = (await self.llm.collective_rpc("ag_install", args=(self.layers, self.groups)))[0]
        self.hook_install_seconds = time.perf_counter() - started
        self.installed = True
        return self

    async def __aexit__(self, *exc):
        if self.llm is not None:
            try:
                if self.installed:
                    await self.llm.collective_rpc("ag_remove")
            finally:
                self.llm.shutdown()
                self.llm = None

    async def warmup(self, requests):
        """Run prompts through the engine, then discard what they contributed; time goes to `warmup_seconds`.

        The first forwards compile and autotune kernels, which would otherwise count as extraction time.
        """
        started = time.perf_counter()
        await self._run(requests)
        await self.llm.collective_rpc("ag_reset")
        self.warmup_seconds += time.perf_counter() - started

    async def extract(self, requests, concurrency=None):
        """Sum the prompts, given as (request ID, group, token IDs); return sums [layers, groups, width], counts."""
        started = time.perf_counter()
        await self._run(requests, concurrency)
        # Tensor-parallel ranks hold identical activations, so rank 0 speaks for all.
        sums, counts = decode_result((await self.llm.collective_rpc("ag_result"))[0])
        self.extract_seconds += time.perf_counter() - started
        return sums, counts

    async def _run(self, requests, concurrency=None):
        requests = list(requests)
        table = {rid: (group, len(tokens)) for rid, group, tokens in requests}
        if len(table) != len(requests):
            raise ValueError("Request IDs must be unique")
        await self.llm.collective_rpc("ag_begin", args=(table,))
        gate = asyncio.Semaphore(concurrency or 2 * self.options["max_num_seqs"])

        async def submit(rid, tokens):
            async with gate:
                async for _ in self.llm.generate({"prompt_token_ids": tokens}, self.params, rid):
                    pass

        tasks = [asyncio.create_task(submit(rid, tokens)) for rid, _, tokens in requests]
        try:
            await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
