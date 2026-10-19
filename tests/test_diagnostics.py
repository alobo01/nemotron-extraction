"""Numerical diagnostics must distinguish computation, identity and transport."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from safetensors.torch import load_file

from nemotron_extraction.diagnostics import TokenDiagnostics, TraceHooks, DiagnosticWorker, runtime_config

sys.path.insert(0,str(Path(__file__).parents[1]/'benchmarks'))
import diagnose_divergence as diagnostic


def ids(n,offset=0):
    return [dict(row_id=f'row-{i:04d}',sequence_id='short' if i<37 else 'long',position=i if i<37 else i-37,token_id=i)
            for i in range(offset,offset+n)]


def test_zero_vectors_are_explicit_and_gates_unchanged():
    h=torch.tensor([[0.,0.],[0.,0.],[1.,0.]])
    v=torch.tensor([[0.,0.],[1.,0.],[-1.,0.]])
    d=TokenDiagnostics();d.add(h,v,ids(3));r=d.result()
    assert r['both_zero_rows']==1 and r['one_zero_rows']==1
    assert r['minimum_cosine']==-1 and not r['passed']
    assert r['normalized_mse']==5 and r['relative_l2']==pytest.approx(5**.5)
    assert r['worst_cosine'][1]['relative_error_undefined']
    json.dumps(r,allow_nan=False)


def test_shard_merge_ranking_and_error_budget_match_full_arrays(tmp_path):
    gen=torch.Generator().manual_seed(8)
    h=torch.randn(75,9,generator=gen);v=h.clone();v[55]*=-1;v[72]+=2
    whole=TokenDiagnostics();whole.add(h,v,ids(75))
    merged=TokenDiagnostics()
    for a,b in ((0,37),(37,68),(68,75)):
        piece=TokenDiagnostics();piece.add(h[a:b],v[a:b],ids(b-a,a));merged.merge(piece)
    assert merged.result()==whole.result()
    assert merged.result()['worst_cosine'][0]['position']==18
    assert merged.result()['worst_error'][0]['row_id'] in ('row-0055','row-0072')
    assert len(merged.samples)<=40
    merged.save_samples(tmp_path/'worst.safetensors')
    assert load_file(str(tmp_path/'worst.safetensors'))
    with pytest.raises(ValueError,match='Duplicate'): merged.merge(whole)


def test_normalization_retains_scale_error_and_rejects_bad_rows():
    h=torch.ones(40,3,dtype=torch.bfloat16);v=h*2
    d=TokenDiagnostics();d.add(h,v,ids(40));r=d.result()
    assert r['normalized_mse']==1 and not r['passed']
    assert r['mean_cosine']==pytest.approx(1,abs=1e-6)
    with pytest.raises(ValueError,match='Nonfinite'):
        TokenDiagnostics().add(h.float(),h.float()*float('nan'),ids(40))
    with pytest.raises(ValueError,match='identity'):
        TokenDiagnostics().add(h,v,ids(39))


class Mixer(nn.Module):
    def forward(self,hidden_states): return hidden_states*2


class Block(nn.Module):
    def __init__(self):
        super().__init__();self.norm=nn.Identity();self.mixer=Mixer()
    def forward(self,hidden_states): return hidden_states+self.mixer(self.norm(hidden_states))


class Model(nn.Module):
    def __init__(self):
        super().__init__();self.model=nn.Module();self.model.embed_tokens=nn.Embedding(64,4)
        self.model.layers=nn.ModuleList([Block(),Block()])
    def forward(self,input_ids):
        x=self.model.embed_tokens(input_ids)
        for layer in self.model.layers: x=layer(x)
        return x


def test_anchor_trace_survives_mutation_replays_common_input_and_cleans_up():
    model=Model();hooks=TraceHooks(model)
    tokens=torch.tensor([[1,2,3],[4,5,6]])
    hooks.span=(3,6)
    model(tokens)
    embedding=hooks.values['embedding'].clone()
    assert embedding.shape==(3,4)
    assert torch.equal(hooks.values['L00_resid_post'],embedding*3)
    with torch.no_grad(): model.model.embed_tokens.weight.zero_()
    assert torch.equal(hooks.values['embedding'],embedding)
    hooks.reset();hooks.span=(0,3)
    hooks.replay=dict(layer=0,operation='mixer',input=torch.ones(3,4))
    model(tokens[:1]);assert torch.equal(hooks.values['L00_mixer_out'],torch.full((3,4),2.))
    hooks.close()
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())


def test_runtime_config_converts_torch_version_to_rpc_safe_builtin_string():
    assert type(runtime_config(Model())['torch_version']) is str


def test_scheduler_records_reordered_duplicates_and_rejects_partial_prefill():
    worker=DiagnosticWorker();worker.diag_trace=SimpleNamespace(span=None)
    worker.diag_case=dict(requests={'a':[1,2],'b':[3,4],'c':[1,2]},anchor='a',scheduler=[],covered=set())
    batch=SimpleNamespace(req_ids=['b-01234567','c-01234567','a-01234567'],
        token_ids_cpu=torch.tensor([[3,4],[1,2],[1,2]]),num_computed_tokens_cpu=[0,0,0])
    worker.model_runner=SimpleNamespace(input_batch=batch,query_start_loc=SimpleNamespace(cpu=torch.tensor([0,2,4,6])))
    worker._diag_inputs(None,(),{})
    assert worker.diag_trace.span==(4,6)
    assert worker.diag_case['covered']=={'a','b','c'}
    assert len(worker.diag_case['scheduler'])==3
    worker.diag_case['covered'].clear();batch.num_computed_tokens_cpu=[1,0,0]
    with pytest.raises(RuntimeError,match='Partial prefill'): worker._diag_inputs(None,(),{})


def test_case_definitions_keep_anchor_identity_and_short_tail():
    rows=torch.arange(64*2048).reshape(64,2048)
    for name,tokens,anchor in diagnostic.cases(rows):
        expected=rows[0,:37] if name=='short37' else rows[0]
        assert torch.equal(tokens[anchor],expected)
    assert diagnostic.boundary_order('L02_mixer_input')<diagnostic.boundary_order('L02_mixer_out')


def test_comparison_selects_first_failure_and_saves_worst_samples(tmp_path):
    from safetensors.torch import save_file
    for engine in ('transformers','vllm'):
        d=tmp_path/engine;d.mkdir()
        h=torch.ones(37,4);v=h if engine=='transformers' else h*2
        save_file({'token_ids':torch.arange(37),'embedding':h.clone(),'L00_resid_pre':h.clone(),
                   'L00_mixer_input':h.clone(),'L00_mixer_out':v.clone(),'L00_resid_post':v.clone()},str(d/'alone.safetensors'))
    diagnostic.analyse(SimpleNamespace(output=tmp_path))
    result=json.loads((tmp_path/'analysis.json').read_text())
    assert result['first_failing_boundary']=='L00_mixer_out'
    assert result['replay']['key']=='L00_mixer_input'
    assert result['gates_unchanged']


def test_original_transformers_persistence_matches_same_forward_oracle_and_reuses_buffers(tmp_path):
    import asyncio
    from nemotron_extraction import HookSite
    class Decoder(nn.Module):
        def __init__(self):
            super().__init__();self.layers=nn.ModuleList([Block()])
        def forward(self,input_ids,use_cache=False):
            return self.layers[0](input_ids.float().unsqueeze(-1).repeat(1,1,4))
    class Wrapper(nn.Module):
        def __init__(self): super().__init__();self.model=Decoder()
    model=Wrapper()
    with diagnostic.IntegrityHFExtractor(model,[HookSite(0,'resid_pre'),HookSite(0,'resid_post')],device='cpu') as extractor:
        rows=torch.tensor([[1,2,3],[4,5,6],[1,2,3]])
        async def collect():
            return [r async for r in extractor.stream_to_disk([rows,rows.flip(0),rows[:1,:2]],tmp_path)]
        receipts=asyncio.run(collect())
        assert all(r['diagnostic_transport_exact'] for r in receipts)
        assert {r['diagnostic_slot'] for r in receipts}=={0,1}
        assert torch.equal(load_file(receipts[-1]['path'])['token_ids'],rows[:1,:2])
        assert sorted(extractor.pool.free)==[0,1]
    assert not model.model._forward_hooks


def test_same_forward_comparison_detects_disagreement_before_transport():
    worker=DiagnosticWorker()
    worker.diag_case={'integrity':True,'equality':[]}
    worker.diag_package={'site':torch.ones(2,3)}
    worker.diag_sibling=SimpleNamespace(sae_captured={'site':torch.zeros(2,3)})
    with pytest.raises(RuntimeError,match='callbacks differ'):
        worker._diag_equal(None,None,None)
