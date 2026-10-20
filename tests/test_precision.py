import hashlib
from pathlib import Path

import pytest
import torch

from nemotron_extraction.precision import checkpoint_source_reference, ieee_precision, precision_settings


def test_ieee_precision_restores_state_after_failure():
    previous = precision_settings()
    with pytest.raises(RuntimeError, match='controlled failure'):
        with ieee_precision() as current:
            assert current['bf16_reduced_reduction'] is False
            if current['api'] == 'modern':
                assert all(current[k] == 'ieee' for k in ('global', 'matmul', 'cudnn', 'conv', 'rnn'))
            else:
                assert not current['matmul_tf32'] and not current['cudnn_tf32']
            raise RuntimeError('controlled failure')
    assert precision_settings() == previous


def test_reference_source_checksum_and_missing_classes_fail_closed(tmp_path):
    source = tmp_path / 'reference.py'
    source.write_text('raise RuntimeError("imports must not execute")\n')
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='checksum'):
        checkpoint_source_reference(source, '0'*64, {})
    with pytest.raises(ValueError, match='class'):
        checkpoint_source_reference(source, digest, {})


def test_source_loader_uses_frozen_classes_without_executing_imports(tmp_path):
    source = tmp_path / 'reference.py'
    source.write_text('''
raise RuntimeError("Mamba imports must not execute")
class NemotronHMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
class NemotronHTopkRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
class NemotronHMOE(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.weight = nn.Parameter(torch.empty(config.hidden_size))
''')
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    model = checkpoint_source_reference(source, digest, dict(hidden_size=7))
    assert model.weight.device.type == 'meta'
    assert model.weight.shape == (7,)
    assert model.layer_idx == 1


def test_final_validation_requires_every_precision_and_probe_check():
    import sys
    sys.path.insert(0, str(Path(__file__).parents[1] / 'benchmarks'))
    from precision_validation import gate_passed
    result = dict(strict_pass=True, exact_router_logits=True,
                  expert_sets_equal=True, persisted_samples_exact=True)
    assert gate_passed(result)
    for key in result:
        failed = dict(result, **{key: False})
        assert not gate_passed(failed)


def test_worst_one_percent_reports_extremes_and_merges_without_changing_ranking():
    from nemotron_extraction.diagnostics import TokenDiagnostics
    a = torch.ones(101, 2)
    b = a.clone()
    b[7] *= -1
    b[100] *= 3
    identities = [dict(row_id=str(i), sequence_id='anchor', position=i, token_id=i) for i in range(101)]
    whole, left, right = TokenDiagnostics(), TokenDiagnostics(), TokenDiagnostics()
    whole.add(a, b, identities)
    left.add(a[:50], b[:50], identities[:50])
    right.add(a[50:], b[50:], identities[50:])
    observed = whole.result()
    assert observed == left.merge(right).result()
    tail = observed['worst_one_percent']
    assert tail['count'] == 2
    assert {r['position'] for r in tail['rows_by_squared_error']} == {7, 100}
    assert tail['error_energy_share'] == 1
    assert observed['maximum_cosine_disagreement'] == pytest.approx(2)
    assert observed['maximum_absolute_error'] == 2
    errors = (b-a).double().square().sum(-1)/a.double().square().sum(-1).mean()
    assert observed['normalized_mse_std'] == pytest.approx(float(errors.std(correction=0)))


def test_reference_accumulation_preserves_expert_order_under_cancellation():
    from nemotron_extraction.moe_diagnostics import manual_nemotron_moe
    x = torch.ones(1, 1)
    up = torch.ones(3, 1, 1)
    down = torch.tensor([1e8, -1e8, 1.]).reshape(3, 1, 1)
    # Slot order gives (1e8 + 1) - 1e8 == 0; expert order gives 1.
    result = manual_nemotron_moe(x, torch.tensor([[0, 2, 1]]), torch.ones(1, 3), up, down)
    assert torch.equal(result['output'], torch.ones(1, 1))
