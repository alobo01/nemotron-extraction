"""Precision controls and checkpoint-source replay, isolated from extraction."""
from __future__ import annotations

import ast
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F
from safetensors import safe_open


def tensor_digest(value):
    return hashlib.sha256(value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def precision_settings():
    # Do not mix the legacy TF32 API with the API introduced after torch 2.9.
    modern = tuple(int(x) for x in str(torch.__version__).split('+')[0].split('.')[:2]) >= (2, 9)
    if modern:
        settings = {name: getattr(owner, 'fp32_precision') for name, owner in (
            ('global', torch.backends), ('matmul', torch.backends.cuda.matmul),
            ('cudnn', torch.backends.cudnn), ('conv', torch.backends.cudnn.conv),
            ('rnn', torch.backends.cudnn.rnn))}
    else:
        settings = dict(matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                        cudnn_tf32=torch.backends.cudnn.allow_tf32)
    settings.update(api='modern' if modern else 'legacy',
        bf16_reduced_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)
    return settings


def set_precision(settings):
    if settings['api'] == 'modern':
        for name, owner in (('global', torch.backends), ('matmul', torch.backends.cuda.matmul),
                            ('cudnn', torch.backends.cudnn), ('conv', torch.backends.cudnn.conv),
                            ('rnn', torch.backends.cudnn.rnn)):
            owner.fp32_precision = settings[name]
    else:
        torch.backends.cuda.matmul.allow_tf32 = settings['matmul_tf32']
        torch.backends.cudnn.allow_tf32 = settings['cudnn_tf32']
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = settings['bf16_reduced_reduction']


def ieee_settings():
    settings = precision_settings()
    if settings['api'] == 'modern':
        settings.update({k: 'ieee' for k in ('global', 'matmul', 'cudnn', 'conv', 'rnn')})
    else:
        settings.update(matmul_tf32=False, cudnn_tf32=False)
    settings['bf16_reduced_reduction'] = False
    return settings


@contextmanager
def ieee_precision():
    previous = precision_settings()
    try:
        set_precision(ieee_settings())
        yield precision_settings()
    finally:
        set_precision(previous)


def checkpoint_source_reference(source, expected_sha256, config):
    """Execute only the three frozen NVIDIA reference MoE class definitions.

    Preserve NVIDIA/Hugging Face source attribution and equations, including
    unsorted top-k and expert-index accumulation order. No Mamba imports or
    alternate Transformers installation are needed for this operator replay.
    """
    source = Path(source)
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError('Reference model source checksum differs from the frozen trace')
    wanted = {'NemotronHMLP', 'NemotronHMOE', 'NemotronHTopkRouter'}
    nodes = [node for node in ast.parse(raw).body if isinstance(node, ast.ClassDef) and node.name in wanted]
    if {node.name for node in nodes} != wanted:
        raise ValueError('Frozen reference source lacks the expected Nemotron operator classes')
    namespace = dict(torch=torch, nn=nn, F=F, Optional=Optional,
                     ACT2FN={'relu2': lambda x: F.relu(x).square()})
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), namespace)
    with torch.device('meta'):
        reference = namespace['NemotronHMOE'](SimpleNamespace(**config), layer_idx=1)
    return reference


class PrecisionWorker:
    """Standalone worker for aligned, layer-1 expert validation only."""

    def precision_prepare(self, model_path, source_path, source_sha256):
        from .hooks import layers_of
        model_path = Path(model_path)
        config = json.loads((model_path / 'config.json').read_text())
        if config['n_group'] != 1 or config['topk_group'] != 1 or config['num_experts_per_tok'] != 6:
            raise ValueError('Expected the frozen one-group, top-six router')
        _, layers = layers_of(self.model_runner.model)
        self.precision_mixer = mixer = layers[1].mixer
        self.precision_reference = reference = checkpoint_source_reference(source_path, source_sha256, config)
        params = dict(mixer.named_parameters())
        up = next(p for n, p in params.items() if n.endswith('w13_weight'))
        down = next(p for n, p in params.items() if n.endswith('w2_weight'))
        if tuple(up.shape) != (config['n_routed_experts'], config['moe_intermediate_size'], config['hidden_size']):
            raise ValueError('Unsupported expert layout')
        index = json.loads((model_path / 'model.safetensors.index.json').read_text())['weight_map']
        verified = []
        prefix = 'backbone.layers.1.mixer.'

        def read(name):
            key = prefix + name
            with safe_open(str(model_path / index[key]), framework='pt') as file:
                return file.get_tensor(key)

        def assign(module, name, value, checkpoint_name, allow_router_cast=False):
            original = read(checkpoint_name)
            if (not allow_router_cast and original.dtype != value.dtype) or not torch.equal(original.to(value.dtype), value.detach().cpu()):
                raise ValueError(f'Loaded parameter differs from checkpoint: {checkpoint_name}')
            verified.append(dict(name=checkpoint_name, checkpoint_dtype=str(original.dtype),
                loaded_dtype=str(value.dtype), checkpoint_sha256=tensor_digest(original)))
            if allow_router_cast:
                delta = original.float() - value.detach().float().cpu()
                verified[-1].update(loaded_cast_changes_values=not torch.equal(original.float(), value.detach().float().cpu()),
                                    maximum_cast_absolute_error=float(delta.abs().max()))
            # The reference router keeps original checkpoint values in float32.
            tensor = original.to(device=value.device, dtype=torch.float32) if allow_router_cast else value.detach()
            setattr(module, name, nn.Parameter(tensor, requires_grad=False))

        for expert in range(config['n_routed_experts']):
            assign(reference.experts[expert].up_proj, 'weight', up[expert], f'experts.{expert}.up_proj.weight')
            assign(reference.experts[expert].down_proj, 'weight', down[expert], f'experts.{expert}.down_proj.weight')
        for projection in ('up_proj', 'down_proj'):
            assign(getattr(reference.shared_experts, projection), 'weight',
                getattr(mixer.shared_experts, projection).weight, f'shared_experts.{projection}.weight')
        assign(reference.gate, 'weight', mixer.gate.weight, 'gate.weight', allow_router_cast=True)
        original_bias = read('gate.e_score_correction_bias')
        if not torch.equal(original_bias, mixer.gate.e_score_correction_bias.detach().cpu()):
            raise ValueError('Router correction bias differs from checkpoint')
        reference.gate.e_score_correction_bias = original_bias.to(up.device)
        verified.append(dict(name='gate.e_score_correction_bias', checkpoint_dtype=str(original_bias.dtype),
                             checkpoint_sha256=tensor_digest(original_bias)))
        reference.eval()
        self.precision_config = self.model_runner.vllm_config
        self.precision_original_gate = mixer.gate.forward
        self.precision_manifest = dict(status='ready', verified_parameters=verified,
            reference_source_sha256=source_sha256, torch=str(torch.__version__), cuda=str(torch.version.cuda),
            package_path=str(Path(__file__).resolve()), gate_original_dtype=str(mixer.gate.weight.dtype),
            reference_gate_dtype=str(reference.gate.weight.dtype),
            native_router_dispatch={name: bool(getattr(mixer.gate, name)) for name in (
                'allow_ll_bf16_gemm', 'allow_dsv3_router_gemm', 'allow_fp32_router_gemm',
                'allow_bf16x3_router_gemm', 'allow_cublas_router_gemm') if hasattr(mixer.gate, name)},
            scope='Layer-1 operator from frozen Transformers source; full-model Mamba execution is not compared')
        return self.precision_manifest

    def precision_case(self, transformer_trace, vllm_trace, length, name, output_dir, reverse_duplicate=False):
        from vllm.forward_context import set_forward_context
        from .diagnostics import TokenDiagnostics, write_json
        reference, mixer = self.precision_reference, self.precision_mixer
        device = reference.gate.weight.device
        traces = []
        tokens = None
        for path in (transformer_trace, vllm_trace):
            with safe_open(str(path), framework='pt') as file:
                ids = file.get_tensor('token_ids')[:length]
                if tokens is not None and not torch.equal(ids, tokens):
                    raise ValueError('Saved trace token IDs differ')
                tokens = ids
                traces.append(file.get_tensor('L01_mixer_input')[:length].to(device))
        if tokens.numel() != length:
            raise ValueError('Saved anchor is shorter than the requested case')
        path = Path(output_dir) / name
        path.mkdir(parents=True, exist_ok=False)
        result = dict(name=name, tokens=length, comparisons={}, strict_pass=True,
                      exact_router_logits=True, expert_sets_equal=True, persisted_samples_exact=True)
        router = mixer.experts.router
        original_selector = router.select_experts
        original_forward = mixer.gate.forward
        routes = {}

        def selector(*args, **kwargs):
            weights, ids = original_selector(*args, **kwargs)
            routes.update(weights=weights.detach().clone(), ids=ids.detach().clone())
            return weights, ids

        def aligned_gate(x):
            logits = F.linear(x.float(), reference.gate.weight)
            routes['logits'] = logits.detach().clone()
            return logits, None

        def compare(label, a, b, row_tokens, positions, required=True):
            a, b = a.detach().cpu(), b.detach().cpu()
            if a.dtype != b.dtype or a.shape != b.shape or not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
                raise ValueError(f'Invalid output: {label}')
            metric = TokenDiagnostics()
            metric.add(a, b, [dict(sequence_id='frozen-anchor', position=int(p), token_id=int(t),
                                   row_id=f'{name}:{label}:{i}') for i, (p, t) in enumerate(zip(positions, row_tokens))])
            summary = metric.result()
            summary['rows_cosine_at_least_0_990'] = sum(row['cosine'] >= .990 for row in metric.rows)
            summary['anchor_99_percent_rule'] = summary['rows_cosine_at_least_0_990'] / len(metric.rows) >= .99
            write_json(path / (label + '.json'), dict(summary=summary, rows=metric.rows))
            samples = path / (label + '.safetensors')
            metric.save_samples(samples)
            # Independently verify the persisted diagnostic tensors are finite.
            with safe_open(str(samples), framework='pt') as file:
                for i, row_id in enumerate(sorted(metric.samples)):
                    for j, side in enumerate(('reference', 'candidate')):
                        restored = file.get_tensor(f'{i}_{side}')
                        expected = metric.samples[row_id][j]
                        if restored.dtype != expected.dtype or restored.shape != expected.shape or not torch.equal(restored, expected):
                            raise ValueError('Persisted diagnostic sample changed')
            result['comparisons'][label] = summary
            summary['required_for_aligned_operator_acceptance'] = required
            if required:
                result['strict_pass'] &= summary['passed']

        try:
            router.select_experts = selector
            with ieee_precision() as settings, torch.inference_mode():
                result['precision'] = settings
                if reverse_duplicate:
                    # Four labelled prompt-equivalent blocks, reversed on the row axis.
                    # This tests the isolated tokenwise expert operation, not the scheduler.
                    x = torch.cat([traces[0], traces[1], traces[0], traces[1]], dim=0).flip(0)
                    cases = [('duplicate_reversed', x, tokens.repeat(4).flip(0),
                              torch.arange(length).repeat(4).flip(0))]
                else:
                    cases = [('transformers_input', traces[0], tokens, torch.arange(length)),
                             ('vllm_input', traces[1], tokens, torch.arange(length))]
                outputs = []
                for label, x, row_tokens, positions in cases:
                    reference_ids, reference_weights = reference.gate(x)
                    reference_logits = F.linear(x.float(), reference.gate.weight)
                    # Warm both complete operator shapes before collecting timed calls.
                    mixer.gate.forward = aligned_gate
                    with set_forward_context(None, self.precision_config, num_tokens=len(x)):
                        mixer(x)
                    reference(x)
                    torch.cuda.synchronize()
                    import time
                    started = time.perf_counter()
                    expected = reference(x)
                    torch.cuda.synchronize()
                    reference_seconds = time.perf_counter() - started
                    started = time.perf_counter()
                    with set_forward_context(None, self.precision_config, num_tokens=len(x)):
                        observed = mixer(x)
                    torch.cuda.synchronize()
                    native_seconds = time.perf_counter() - started
                    result['exact_router_logits'] &= torch.equal(reference_logits, routes['logits'])
                    result['expert_sets_equal'] &= torch.equal(reference_ids.sort(-1).values, routes['ids'].long().sort(-1).values)
                    compare(label, expected, observed, row_tokens, positions)
                    result['comparisons'][label].update(reference_operator_seconds=reference_seconds,
                        aligned_vllm_operator_seconds=native_seconds, actual_forward_tokens=len(x))
                    outputs.append((expected.detach().cpu(), observed.detach().cpu()))
                    if not result['exact_router_logits'] or not result['expert_sets_equal']:
                        result['strict_pass'] = False
                if not reverse_duplicate:
                    # Different inherited inputs remain a separate scientific question.
                    # Never label same-input agreement as full-model equivalence.
                    compare('inherited_input_effect', outputs[0][0], outputs[1][0], tokens, torch.arange(length), required=False)
                    compare('aligned_cross_native_inputs', outputs[0][1], outputs[1][1], tokens, torch.arange(length), required=False)
                    for label, trace_path, actual in (
                        ('reference_vs_saved_transformers', transformer_trace, outputs[0][0]),
                        ('aligned_vs_saved_vllm', vllm_trace, outputs[1][1])):
                        with safe_open(str(trace_path), framework='pt') as file:
                            saved = file.get_tensor('L01_mixer_out')[:length]
                        compare(label, saved, actual, tokens, torch.arange(length), required=False)
        finally:
            mixer.gate.forward = original_forward
            router.select_experts = original_selector
        result.update(torch=str(torch.__version__), cuda=str(torch.version.cuda))
        write_json(path / 'receipt.json', result)
        return result

    def precision_close(self):
        if hasattr(self, 'precision_mixer'):
            self.precision_mixer.gate.forward = self.precision_original_gate
            self.precision_reference = None
        return True
