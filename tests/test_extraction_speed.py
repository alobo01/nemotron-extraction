from pathlib import Path
import sys

import pytest
import torch
from safetensors.torch import save_file

sys.path.insert(0,str(Path(__file__).parents[1]/'benchmarks'))
import extraction_speed as speed
from nemotron_extraction import HookSite,HFExtractor,VLLMExtractor


def test_workload_preserves_all_tokens_and_full_shapes():
    rows=torch.arange(64*2048).reshape(64,2048)
    for size in (8,64):
        batches=speed.workload(rows,size)
        assert all(b.shape==(size,2048) for b in batches)
        assert torch.equal(torch.cat(batches),rows)
    with pytest.raises(ValueError): speed.workload(rows,32)


def test_comparison_aligns_different_shard_boundaries_and_hashes(tmp_path):
    rows=torch.arange(4*2048).reshape(4,2048)
    site=HookSite(0,'resid_post')
    measurements=[]
    for name,size in (('left',1),('right',4)):
        receipts=[]
        for i,tokens in enumerate(rows.split(size)):
            path=tmp_path/f'{name}-{i}.safetensors'
            save_file({'token_ids':tokens,site.key:tokens.reshape(-1,1).float()},str(path))
            receipts.append(dict(path=str(path),tokens=tokens.numel()))
        measurements.append(dict(receipts=receipts))
    result=speed.compare_measurements(*measurements,rows,[site],tmp_path,tmp_path/'metrics')[site.key]
    assert result['exact'] and result['passed'] and result['normalized_mse']==0
    assert result['reference_sha256']==result['candidate_sha256']
    speed.prune_receipts(measurements[1],tmp_path)
    assert not Path(measurements[1]['receipts'][0]['path']).exists()
    assert Path(measurements[0]['receipts'][0]['path']).exists()


def test_pruning_rejects_paths_outside_this_configuration(tmp_path):
    outside=tmp_path/'outside.safetensors';outside.write_bytes(b'keep')
    with pytest.raises(ValueError,match='outside'):
        speed.prune_receipts(dict(receipts=[dict(path=str(outside))]),tmp_path/'run')
    assert outside.exists()


@pytest.mark.parametrize('factory',[HFExtractor,VLLMExtractor])
@pytest.mark.parametrize('threads',[0,3,True])
def test_writer_limit_rejected_before_loading(factory,threads):
    with pytest.raises(ValueError,match='publication_threads'):
        factory('unused',[HookSite(0,'resid_post')],publication_threads=threads)
