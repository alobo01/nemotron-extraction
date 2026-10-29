import hashlib
from concurrent.futures import ThreadPoolExecutor
import pytest
import torch
from safetensors.torch import load_file
from nemotron_extraction.transport import publish_site_directory


def test_site_publication_matches_buffers_and_checksums(tmp_path):
    tensors = {"token_ids": torch.tensor([[1, 2]]), "L2_resid_post": torch.arange(6).reshape(2, 3).float()}
    with ThreadPoolExecutor(max_workers=8) as pool:
        receipt = publish_site_directory(tensors, tmp_path / "wave", pool)
    assert receipt["version"] == 2 and receipt["finite_validation"] == "device"
    assert receipt["bytes"] == sum(p.stat().st_size for p in (tmp_path / "wave").iterdir())
    for key, tensor in tensors.items():
        record = receipt["files"][key]
        assert torch.equal(load_file(record["path"])[key], tensor)
        assert record["sha256"] == hashlib.sha256(memoryview(tensor.view(torch.uint8).numpy())).hexdigest()
    assert not list(tmp_path.glob("*.partial"))
    with ThreadPoolExecutor() as pool, pytest.raises(FileExistsError):
        publish_site_directory(tensors, tmp_path / "wave", pool)


def test_site_write_failure_never_publishes_partial_directory(tmp_path, monkeypatch):
    import nemotron_extraction.transport as transport

    def fail(*a, **kw):
        raise OSError("disk full")

    monkeypatch.setattr(transport, "save_file", fail)
    with ThreadPoolExecutor(max_workers=2) as pool, pytest.raises(OSError, match="disk full"):
        publish_site_directory({"token_ids": torch.ones(2, 2)}, tmp_path / "wave", pool)
    assert not list(tmp_path.iterdir())


def test_failed_device_check_prevents_any_publication(tmp_path):
    from nemotron_extraction.worker import ActivationHookExtension
    from concurrent.futures import Future

    copied = Future()
    copied.set_result(0.0)
    wave = dict(
        copies=[(copied, 1.0)],
        forward_events=[],
        origin_cpu=0,
        layout="per_site",
        finite_flags=[("L2_resid_post", torch.tensor(False))],
    )
    with pytest.raises(ValueError, match="nonfinite activations"):
        ActivationHookExtension._nh_publish(wave)
    assert not list(tmp_path.iterdir())


def test_vllm_per_site_path_is_a_directory_without_loading_model(tmp_path):
    from nemotron_extraction.vllm import VLLMExtractor

    extractor = VLLMExtractor.__new__(VLLMExtractor)
    extractor.publication_layout = "per_site"
    assert extractor.shard_path(tmp_path, 3) == tmp_path / "000003"
    extractor.publication_layout = "monolithic"
    assert extractor.shard_path(tmp_path, 3) == tmp_path / "000003.safetensors"
