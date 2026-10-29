import pytest

from nemotron_extraction.batching import AutoBatchConfig


@pytest.mark.parametrize("gib", [80, 94, 141])
def test_device_fit_alignment_and_budget(gib):
    c = AutoBatchConfig(
        resident_bytes=56 * 1024**3,
        reference_peak_bytes=84 * 1024**3,
        reference_sequences=64,
        activation_bytes_per_token=83712,
    )
    n = c.fit(gib * 1024**3)
    assert n >= 4 and n % 4 == 0
    assert (
        c.resident_bytes + c.workspace_reserve_bytes + n * (c.reference_peak_bytes - c.resident_bytes) / 64
        <= 0.96 * gib * 1024**3
    )
    assert n * 2048 * 83712 <= c.publication_budget_bytes
    assert 2 * n * 2048 * 83712 <= c.host_capture_budget_bytes


def test_more_memory_fits_more_sequences_and_external_allocations_reduce_fit():
    c = AutoBatchConfig(
        resident_bytes=56 * 1024**3,
        reference_peak_bytes=84 * 1024**3,
        reference_sequences=64,
        activation_bytes_per_token=53760,
        host_capture_budget_bytes=128 * 1024**3,
        publication_budget_bytes=64 * 1024**3,
    )
    assert c.fit(141 * 1024**3) > c.fit(94 * 1024**3) > c.fit(80 * 1024**3)
    assert c.fit(94 * 1024**3, external_used_bytes=8 * 1024**3) < c.fit(94 * 1024**3)


def test_no_frame_fit_is_an_error():
    c = AutoBatchConfig(56 * 1024**3, 84 * 1024**3, 64, 83712)
    with pytest.raises(MemoryError):
        c.fit(40 * 1024**3)
