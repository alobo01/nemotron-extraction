# /// script
# requires-python = ">=3.10"
# dependencies = ["nemotron-extraction[hf,vllm,benchmark]==1.1.0"]
# ///
"""Bounded numerical investigation of where the two backends diverge.

uv run --no-project --with /path/to/wheel.whl diagnose_divergence.py --help
Run inspect, backend/integrity, analyse, then backend --replay-selection and
analyse again. Infrastructure and submission settings are supplied externally.
Diagnostic synchronization invalidates production throughput comparisons.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys
import time
import traceback

import torch
from safetensors.torch import load_file, save_file
from nemotron_extraction import HFExtractor, VLLMExtractor
from benchlib.common import digest, write_json
from benchlib.stats import TokenDiagnostics
from benchlib.trace import TraceHooks, runtime_config
from nemotron_extraction.hooks import repair_time_step_limits
from nemotron_extraction.transport import SLOTS
from targeted_validation import Utilisation

MODEL='nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16'
REVISION='fe4143e388795a58a92974891b2e45092a53dc1d'
ANCHOR_SHA256='c5e6f8311596eba310ae9c0f181334dc5d2ae8fcbd8290d7f26ce995b30fb840'


def software():
    result={}
    for name in ('torch','transformers','vllm','mamba-ssm','causal-conv1d','nemotron-extraction'):
        try: result[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: result[name]=None
    result.update(python=sys.version,cuda=torch.version.cuda)
    return result


def inputs(work):
    manifest=json.loads((work/'manifest.json').read_text())
    if manifest['model']!=MODEL or manifest['model_revision']!=REVISION or manifest['dtype']!='bfloat16':
        raise ValueError('Expected pinned official BF16 checkpoint')
    path=work/'en.safetensors'
    if digest(path)!=manifest['languages']['en']['token_sha256']:
        raise ValueError('Frozen token checksum differs')
    tokens=load_file(str(path))['token_ids']
    if tokens.numel()<64*2048:
        raise ValueError('Need first 64 frozen English sequences')
    return manifest,tokens[:64*2048].reshape(64,2048)


def cases(rows,pilot=False,names=None):
    selected=[('short128',rows[:1,:128].contiguous(),0),('alone',rows[:1],0)]
    if not pilot:
        for size in (8,64):
            selected += [(f'b{size}',rows[:size],0),(f'b{size}_reversed',rows[:size].flip(0),size-1)]
        selected += [('duplicates',rows[[0,2,0,1]],0),('short37',rows[:1,:37].contiguous(),0)]
    if names:
        unknown=set(names)-{name for name,_,_ in selected}
        if unknown: raise ValueError('Unknown diagnostic cases: '+str(sorted(unknown)))
        selected=[item for item in selected if item[0] in names]
    return selected


def selected_cases(args,rows,spec=None):
    names=getattr(args,'cases',None)
    if spec: names=[spec.get('case','alone')]
    return cases(rows,args.pilot or bool(spec),names)


def identities(tokens,sequence='anchor'):
    return [dict(row_id=f'{sequence}:{i:06d}',sequence_id=sequence,position=i,token_id=int(token))
            for i,token in enumerate(tokens.tolist())]


def boundary_order(key):
    if key=='embedding': return (-1,0)
    prefix,_,kind=key.partition('_')
    return (int(prefix[1:]),{'resid_pre':0,'mixer_input':1,'attn_prelinear':2,'mixer_out':3,'resid_post':4}.get(kind,99))


def inspect_inputs(args):
    manifest,_=inputs(args.work)
    weights=Path(manifest['model_path'])
    index=weights/'model.safetensors.index.json'
    shards=sorted(set(json.loads(index.read_text())['weight_map'].values()))
    if not all((weights/name).is_file() and (weights/name).stat().st_size for name in shards):
        raise ValueError('Weight shards missing')
    import nemotron_extraction
    package=Path(nemotron_extraction.__file__).parent
    record=dict(model=MODEL,model_revision=REVISION,manifest_sha256=digest(args.work/'manifest.json'),
        english_tokens_sha256=manifest['languages']['en']['token_sha256'],script_sha256=digest(__file__),
        package_source_sha256={p.name:digest(p) for p in sorted(package.glob('*.py'))},
        sibling_source_sha256={p.name:digest(p) for p in sorted(args.reference.glob('*.py'))},
        model_config=json.loads((weights/'config.json').read_text()),weight_index_sha256=digest(index),
        weight_shards={name:(weights/name).stat().st_size for name in shards},
        command=sys.argv,software=software(),gates=dict(mean_cosine=.9999,minimum_cosine=.9990,relative_l2=.02),
        synchronization='Diagnostics intentionally synchronize; timings are not extraction throughput')
    target=args.output/'provenance.json'
    if target.exists():
        previous=json.loads(target.read_text())
        for key in ('model_revision','manifest_sha256','script_sha256','package_source_sha256','sibling_source_sha256'):
            if record[key]!=previous[key]: raise ValueError('Provenance changed: '+key)
    else: write_json(target,record)
    return record


def replay_spec(args):
    if not args.replay_selection: return None
    result=json.loads(args.replay_selection.read_text())['replay']
    if result is None: raise ValueError('No operation selected for replay')
    return result


def transformers_backend(args,rows,manifest,destination,monitor):
    from transformers import AutoModelForCausalLM
    started=time.perf_counter()
    model=AutoModelForCausalLM.from_pretrained(manifest['model_path'],torch_dtype=torch.bfloat16,
        device_map='cuda',trust_remote_code=True,local_files_only=True)
    repair_time_step_limits(model)
    model.eval()
    result=dict(startup_seconds=time.perf_counter()-started,runtime=runtime_config(model),cases={})
    decoder=getattr(model,'model',model)
    trace=TraceHooks(model)
    spec=replay_spec(args)
    if spec:
        trace.replay=dict(layer=spec['layer'],operation=spec['operation'],input=load_file(spec['path'])[spec['key']])
        trace.details(spec['layer'])
    try:
        for name,tokens,anchor in selected_cases(args,rows,spec):
            path=destination/(name+'.safetensors');receipt=destination/(name+'.json')
            if receipt.exists():
                r=json.loads(receipt.read_text());result['cases'][name]=r
                if r['status']=='complete' and not path.is_file(): raise ValueError('Missing completed trace')
                continue
            monitor.phase=name
            trace.reset();trace.span=(anchor*tokens.shape[1],(anchor+1)*tokens.shape[1])
            begun=time.perf_counter()
            record=dict(status='running',batch_size=len(tokens),sequence_length=tokens.shape[1],anchor=anchor)
            try:
                with torch.inference_mode(): decoder(input_ids=tokens.to('cuda'),use_cache=False)
                torch.cuda.synchronize()
                values={k:v.cpu().contiguous() for k,v in trace.values.items()}
                values['token_ids']=tokens[anchor].clone()
                if not values or 'embedding' not in values: raise ValueError('Missing embedding trace')
                if not all(bool(torch.isfinite(v).all()) for v in values.values()): raise ValueError('Nonfinite trace')
                partial_path=path.with_suffix('.safetensors.partial')
                save_file(values,str(partial_path));partial_path.replace(path)
                restored=load_file(str(path))
                if not all(torch.equal(v,restored[k]) for k,v in values.items()):
                    raise ValueError('Trace persistence mismatch')
                record.update(status='complete',path=str(path),topology=trace.topology,
                    trace_persistence_exact=True,actual_forward_batches=[len(tokens)],
                    operator_observability=trace.detail_observability)
            except torch.OutOfMemoryError as error:
                record.update(status='memory_failure',error=str(error));torch.cuda.empty_cache()
            record['diagnostic_wall_seconds']=time.perf_counter()-begun
            record['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
            write_json(receipt,record);result['cases'][name]=record
    finally: trace.close()
    return result


async def vllm_backend(args,rows,manifest,destination,monitor):
    from vllm import AsyncEngineArgs,SamplingParams
    from vllm.v1.engine.async_llm import AsyncLLM
    started=time.perf_counter()
    config=dict(model=manifest['model_path'],dtype='bfloat16',trust_remote_code=True,tensor_parallel_size=1,
        enforce_eager=True,enable_prefix_caching=False,enable_chunked_prefill=not args.no_chunked_prefill,
        max_model_len=2049,max_num_seqs=64,max_num_batched_tokens=64*2049,
        skip_tokenizer_init=True,gpu_memory_utilization=.96,
        worker_extension_cls='benchlib.workers.DiagnosticWorker')
    engine=AsyncLLM.from_engine_args(AsyncEngineArgs(**config))
    result=dict(startup_seconds=time.perf_counter()-started,engine_configuration=config,cases={})
    installed=False
    try:
        install_start=time.perf_counter()
        result['runtime']=(await engine.collective_rpc('diag_install',args=(str(args.reference),)))[0]
        installed=True
        result['hook_install_seconds']=time.perf_counter()-install_start
        spec=replay_spec(args)
        requests=selected_cases(args,rows,spec)
        if args.integrity:
            requests=[('duplicate_reordered',rows[[2,0,2,1],:128].contiguous(),1),('short37',rows[:1,:37].contiguous(),0)]
        for name,tokens,anchor in requests:
            path=destination/(name+'.safetensors');receipt=destination/(name+'.json')
            if receipt.exists():
                r=json.loads(receipt.read_text());result['cases'][name]=r
                if r['status']=='complete' and not path.is_file(): raise ValueError('Missing completed trace')
                continue
            monitor.phase=name
            ids=[f'diagnostic-{name}-{i}' for i in range(len(tokens))]
            # Deliberately reverse admission; scheduler receipts establish actual order.
            begun=time.perf_counter()
            await engine.collective_rpc('diag_begin',args=(name,ids,tokens.tolist(),anchor,str(path),args.integrity,spec))
            async def request(index):
                async for _ in engine.generate({'prompt_token_ids':tokens[index].tolist()},SamplingParams(max_tokens=1,temperature=0),ids[index]):
                    pass
            await asyncio.gather(*(request(i) for i in reversed(range(len(tokens)))))
            record=(await engine.collective_rpc('diag_finish'))[0]
            record.update(status='complete',batch_size=len(tokens),sequence_length=tokens.shape[1],anchor=anchor,
                          diagnostic_wall_seconds=time.perf_counter()-begun)
            write_json(receipt,record);result['cases'][name]=record
    finally:
        try:
            if installed: await engine.collective_rpc('diag_close')
        finally: engine.shutdown()
    return result


def pair_metrics(reference,candidate):
    if reference.shape!=candidate.shape: raise ValueError('Replay tensor shapes differ')
    a,b=reference.float(),candidate.float()
    an,bn=a.norm(dim=-1),b.norm(dim=-1)
    cosine=((a*b).sum(-1)/(an*bn).clamp_min(torch.finfo(torch.float32).tiny)).clamp(-1,1)
    cosine[(an==0)&(bn==0)]=1
    error=(b-a).double().square().sum()
    norm=a.double().square().sum()
    nmse=float(error/norm) if norm else (0. if not error else None)
    return {
        'rows':len(a),'mean_cosine':float(cosine.mean()),'minimum_cosine':float(cosine.min()),
        'normalized_mse':nmse,'relative_l2':nmse**.5 if nmse is not None else None,
        'rows_cosine_at_least_0_990':int((cosine>=.990).sum()),
        'fraction_cosine_at_least_0_990':float((cosine>=.990).float().mean()),
        'passes_anchor_99_percent_rule':bool((cosine>=.990).float().mean()>=.99),
        'strict_mean_cosine_pass':bool(cosine.mean()>=.9999),
        'strict_minimum_cosine_pass':bool(cosine.min()>=.9990),
        'strict_relative_l2_pass':bool(nmse is not None and nmse**.5<=.02),
    }


def relative_effect(effect,baseline):
    denominator=float(torch.linalg.vector_norm(baseline.float()))
    return float(torch.linalg.vector_norm(effect.float()))/denominator if denominator else None


async def moe_replay(args):
    from vllm import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM
    manifest,rows=inputs(args.work)
    anchor_hash=hashlib.sha256(rows[0].cpu().numpy().astype('<i8',copy=False).tobytes()).hexdigest()
    if anchor_hash!=ANCHOR_SHA256:
        raise ValueError(f'Frozen anchor checksum differs: {anchor_hash}')
    for trace in (args.transformers_trace,args.vllm_trace):
        if not trace.is_file(): raise FileNotFoundError(trace)
    args.output.mkdir(parents=True,exist_ok=True)
    engine_config=dict(model=manifest['model_path'],dtype='bfloat16',trust_remote_code=True,
        tensor_parallel_size=1,enforce_eager=True,enable_prefix_caching=False,
        enable_chunked_prefill=False,max_model_len=2049,max_num_seqs=1,
        max_num_batched_tokens=2049,skip_tokenizer_init=True,gpu_memory_utilization=.96,
        worker_extension_cls='benchlib.workers.DiagnosticWorker')
    engine=AsyncLLM.from_engine_args(AsyncEngineArgs(**engine_config))
    output=args.output/'layer1-moe-replay.safetensors'
    try:
        worker=(await engine.collective_rpc('diag_moe_replay',args=(str(args.transformers_trace),
            str(args.vllm_trace),str(output),1)))[0]
    finally: engine.shutdown()
    tensors=load_file(str(output))
    if not torch.equal(tensors['token_ids'],rows[0]):
        raise ValueError('Diagnostic traces do not identify frozen anchor sequence 0')
    summary={
        'status':'complete','worker':worker,'engine_configuration':engine_config,
        'source':{'model':manifest['model'],'model_revision':manifest['model_revision'],
            'tokenizer_revision':manifest['tokenizer_revision'],
            'token_file_sha256':manifest['languages']['en']['token_sha256'],
            'anchor_token_ids_i64le_sha256':anchor_hash,
            'package_script_sha256':digest(__file__)},
        'software':software(),
        'saved_cross_backend':pair_metrics(tensors['saved_transformers_output'],tensors['saved_vllm_output']),
        'manual_replay_vs_saved_transformers':pair_metrics(tensors['saved_transformers_output'],tensors['manual_on_transformers_input']),
        'native_replay_vs_saved_vllm':pair_metrics(tensors['saved_vllm_output'],tensors['native_on_vllm_input']),
        'manual_vs_native_on_transformers_input':pair_metrics(tensors['manual_on_transformers_input'],tensors['native_on_transformers_input']),
        'manual_vs_native_on_vllm_input':pair_metrics(tensors['manual_on_vllm_input'],tensors['native_on_vllm_input']),
        'route_matched_manual_vs_native_on_transformers_input':pair_metrics(tensors['route_matched_manual_on_transformers_input'],tensors['native_on_transformers_input']),
        'route_matched_manual_vs_native_on_vllm_input':pair_metrics(tensors['route_matched_manual_on_vllm_input'],tensors['native_on_vllm_input']),
        'tf32_router_different_expert_rows':int((torch.sort(tensors['route_ids_tf32_off'],dim=-1).values
            !=torch.sort(tensors['route_ids_tf32_on'],dim=-1).values).any(dim=-1).sum()),
        'tf32_router_rows':int(tensors['route_ids_tf32_off'].shape[0]),
        'tf32_router_maximum_logit_absolute_difference':float((tensors['router_logits_tf32_off'].float()
            -tensors['router_logits_tf32_on'].float()).abs().max()),
        'router_selection_and_margin':worker.get('manual_router_vs_vllm'),
        'selected_expert_margin':worker.get('selected_expert_margin'),
        'decomposition':{
            'normalization':'Frobenius norm divided by saved Transformers output Frobenius norm',
            'input_effect_relative_norm':relative_effect(tensors['input_effect'],tensors['saved_transformers_output']),
            'implementation_effect_on_candidate_input_relative_norm':relative_effect(tensors['implementation_effect_on_candidate_input'],tensors['saved_transformers_output']),
            'implementation_effect_on_reference_input_relative_norm':relative_effect(tensors['implementation_effect_on_reference_input'],tensors['saved_transformers_output']),
            'interaction_effect_relative_norm':relative_effect(tensors['interaction_effect'],tensors['saved_transformers_output']),
            'additive_reconstruction_within_roundoff':torch.allclose(
                (tensors['native_on_vllm_input']-tensors['manual_on_transformers_input']),
                (tensors['manual_on_vllm_input']-tensors['manual_on_transformers_input'])+
                (tensors['native_on_vllm_input']-tensors['manual_on_vllm_input']),atol=1e-7,rtol=1e-6),
        },
        'numerical_fix_established':False,
        'output_path':str(output),
        'limitations':['Manual expert equations use observed vLLM routes and loaded vLLM weights; they isolate expert execution/accumulation, not the separately loaded Transformers router or weight loader.','The TF32 switch is a controlled vLLM GateLinear call in the replay runtime; verify effective kernel precision from runtime/source receipts before attributing changes to TF32.'],
    }
    write_json(args.output/'layer1-moe-replay.json',summary)
    lines=['# Layer 1 MoE anchor replay','','This replay separates the effect of the saved normalized input from the expert implementation. It uses the loaded vLLM BF16 weights for both the transparent PyTorch equations and the fused vLLM operation; the original Transformers and vLLM traces remain the native-output checks.','',
        '| Comparison | Mean cosine | Minimum cosine | Normalized MSE | Rows ≥ 0.990 |','|---|---:|---:|---:|---:|']
    for label,key in [('Saved cross-backend','saved_cross_backend'),('Manual replay vs saved Transformers','manual_replay_vs_saved_transformers'),('Native replay vs saved vLLM','native_replay_vs_saved_vllm'),('Route-matched manual vs native on Transformers input','route_matched_manual_vs_native_on_transformers_input'),('Route-matched manual vs native on vLLM input','route_matched_manual_vs_native_on_vllm_input')]:
        item=summary[key];lines.append(f"| {label} | {item['mean_cosine']:.8f} | {item['minimum_cosine']:.8f} | {item['normalized_mse']:.8g} | {item['rows_cosine_at_least_0_990']}/{item['rows']} |")
    lines+=['',f"The anchor-specific 99%/0.990 rule {'passes' if summary['saved_cross_backend']['passes_anchor_99_percent_rule'] else 'fails'} on the saved cross-backend outputs. The original strict mean, minimum and relative-L2 gates remain reported above.",'',f"TF32 router control changed expert sets for {summary['tf32_router_different_expert_rows']}/{summary['tf32_router_rows']} rows; maximum logit difference was {summary['tf32_router_maximum_logit_absolute_difference']:.8g}.",'','The decomposition is diagnostic only. It does not claim numerical equivalence or a fix. See the JSON receipt for software/configuration, effect norms, and limitations.']
    (args.output/'layer1-moe-replay.md').write_text('\n'.join(lines)+'\n')
    return summary


class IntegrityExtractor(VLLMExtractor):
    _worker_extension_cls='benchlib.workers.IntegrityWorker'


class IntegrityHFExtractor(HFExtractor):
    """Same-forward oracle around the unchanged Transformers persistence path."""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.oracles={}
        decoder=getattr(self.model,'model',self.model)
        def oracle(module,args,output):
            self.oracles[self.diag_slot]={k:v.detach().reshape(-1,v.shape[-1]).clone().cpu() for k,v in self.captured.items()}
        self.handles.append(decoder.register_forward_hook(oracle))

    def _capture(self,tokens,slot):
        self.diag_slot=slot
        return super()._capture(tokens,slot)

    def _publish(self,tokens,slot,capture,path,index):
        oracle=self.oracles.pop(slot)
        occupied=SLOTS-len(self.pool.free)
        time.sleep(.5)  # hold this slot so the next capture fills the other buffer
        result=super()._publish(tokens,slot,capture,path,index)
        saved=load_file(result['path'])
        exact=torch.equal(saved['token_ids'],tokens) and all(torch.equal(saved[k],v) for k,v in oracle.items())
        if not exact: raise RuntimeError('Transformers asynchronous transfer/persistence mismatch')
        result.update(diagnostic_transport_exact=True,diagnostic_slot=slot,diagnostic_occupied_slots=occupied)
        return result


def package_integrity(args,rows,manifest,destination,monitor):
    site_path=args.output/'sites.json'
    if not site_path.exists():
        raise ValueError('Run Transformers backend first to resolve integrity sites')
    from nemotron_extraction import HookSite
    sites=[HookSite(**s) for s in json.loads(site_path.read_text())]
    started=time.perf_counter()
    extractor=(IntegrityHFExtractor(manifest['model_path'],sites,local_files_only=True) if args.integrity=='transformers'
               else IntegrityExtractor(manifest['model_path'],sites,max_model_len=2049,sequences_per_call=4,
                    enable_chunked_prefill=True,gpu_memory_utilization=.96,sink_dir=str(destination)))
    with extractor:
        result=dict(startup_seconds=time.perf_counter()-started)
        monitor.phase='two_pending_batches'
        batch=rows[[2,0,2,1],:128].contiguous()
        async def collect():
            return [r async for r in extractor.stream_to_disk([batch,batch.flip(0),rows[:1,:37].contiguous()],destination/'shards')]
        receipts=asyncio.run(collect())
        if not all(r['diagnostic_transport_exact'] for r in receipts): raise ValueError('Transfer mismatch')
        occupied=max(r['diagnostic_occupied_slots'] for r in receipts)
        if occupied!=2: raise ValueError('Two pending buffer slots were not exercised')
        result.update(receipts=receipts,two_slots_observed=True,actual_forward_batches=extractor.actual_forward_batches)
    return result


def backend(args):
    torch.set_num_threads(4);torch.manual_seed(0)
    inspect_inputs(args)
    manifest,rows=inputs(args.work)
    variant=args.label or args.backend
    if args.integrity: variant+='-'+args.integrity
    if args.no_chunked_prefill: variant+='-no-chunked'
    if args.replay_selection: variant+='-replay'
    if args.pilot: variant+='-pilot'
    destination=args.output/variant;destination.mkdir(parents=True,exist_ok=True)
    receipt=destination/'result.json'
    if receipt.exists():
        previous=json.loads(receipt.read_text())
        if previous['status']=='complete': print('Preserving complete stage',variant,flush=True);return 0
        raise FileExistsError('Preserve failed result; use a new output directory')
    record=dict(status='running',backend=args.backend,software=software(),command=sys.argv,
        provenance_sha256=digest(args.output/'provenance.json'),gpu=torch.cuda.get_device_name(0))
    monitor=Utilisation(destination/'utilisation.jsonl')
    begun=time.perf_counter()
    try:
        with monitor:
            if args.integrity in ('package','transformers'): result=package_integrity(args,rows,manifest,destination,monitor)
            elif args.backend=='transformers':
                result=transformers_backend(args,rows,manifest,destination,monitor)
                # The same pinned topology also defines the 18 original integrity sites.
                config=json.loads((Path(manifest['model_path'])/'config.json').read_text())
                kinds={'M':'mamba_out','*':'attn_out','E':'moe_out'}
                pattern=config['hybrid_override_pattern']
                sites=[]
                for kind in ('resid_pre','resid_post','mamba_out','moe_out','attn_out','attn_out_prelinear'):
                    eligible=[i for i,c in enumerate(pattern) if kind.startswith('resid_') or kinds.get(c)==('attn_out' if kind=='attn_out_prelinear' else kind)]
                    if not eligible: raise ValueError('Unknown checkpoint topology for '+kind)
                    sites += [dict(layer=i,kind=kind) for i in sorted({eligible[0],eligible[(len(eligible)-1)//2],eligible[-1]})]
                write_json(args.output/'sites.json',sites)
            else: result=asyncio.run(vllm_backend(args,rows,manifest,destination,monitor))
            record.update(result,status='complete')
    except Exception as error:
        message=str(error)
        record.update(status='memory_failure' if isinstance(error,torch.OutOfMemoryError) or 'out of memory' in message.lower() else 'error',
            error_type=type(error).__name__,error=message,traceback=traceback.format_exc())
    record.update(utilisation=monitor.summary(),process_wall_seconds=time.perf_counter()-begun)
    write_json(receipt,record)
    print(json.dumps({k:record.get(k) for k in ('status','error','startup_seconds','process_wall_seconds')}),flush=True)
    return 0 if record['status']=='complete' else 1


def compare_traces(left_path,right_path,output):
    left,right=load_file(str(left_path)),load_file(str(right_path))
    if not torch.equal(left['token_ids'],right['token_ids']): raise ValueError('Anchor token IDs differ')
    ids=identities(left['token_ids']);result={}
    # Detail module names alone do not establish corresponding mathematical roles.
    common=[k for k in left.keys() & right.keys() if k!='token_ids' and '_raw_' not in k and '_detail_' not in k]
    for key in sorted(common,key=boundary_order):
        metric=TokenDiagnostics();metric.add(left[key],right[key],ids)
        result[key]=metric.result()
        metric.save_samples(output/(key+'.safetensors'))
        write_json(output/(key+'.json'),dict(summary=result[key],rows=metric.rows,
            position_blocks=[dict(start=start,stop=min(start+128,len(metric.rows)),
                mean_cosine=sum(r['cosine'] for r in metric.rows[start:start+128])/len(metric.rows[start:start+128]),
                normalized_mse=(sum(r['error_squared'] for r in metric.rows[start:start+128])/
                    sum(r['reference_squared'] for r in metric.rows[start:start+128]))
                    if sum(r['reference_squared'] for r in metric.rows[start:start+128]) else None)
                for start in range(0,len(metric.rows),128)]))
    if not common: raise ValueError('No common trace boundaries')
    return result


def analyse(args):
    analysis_started=time.perf_counter()
    args.output.mkdir(parents=True,exist_ok=True)
    directory=args.output/'analysis';directory.mkdir(exist_ok=True)
    result=dict(observed={},supported_inference=[],unresolved=[],gates_unchanged=True)
    for backend_name in ('transformers','vllm'):
        base=args.output/backend_name/'alone.safetensors'
        if not base.exists(): continue
        for name in ('b8','b64','b8_reversed','b64_reversed','duplicates'):
            path=args.output/backend_name/(name+'.safetensors')
            if path.exists():
                samples=directory/(backend_name+'-'+name);samples.mkdir(exist_ok=True)
                result['observed'][backend_name+'-batch-'+name]=compare_traces(base,path,samples)
    def trace_path(engine,name):
        for label in (engine,engine+'-pilot'):
            path=args.output/label/(name+'.safetensors')
            if path.exists(): return path
        return args.output/engine/(name+'.safetensors')
    primary='short128' if all(trace_path(b,'short128').exists() for b in ('transformers','vllm')) else 'alone'
    left=trace_path('transformers',primary);right=trace_path('vllm',primary)
    result['primary_case']=primary
    result['replay']=None
    if left.exists() and right.exists():
        samples=directory/'cross-backend';samples.mkdir(exist_ok=True)
        cross=compare_traces(left,right,samples);result['observed']['cross_backend']=cross
        for name in ('short128','alone','b8','b64','b8_reversed','b64_reversed','duplicates','short37'):
            a=trace_path('transformers',name);b=trace_path('vllm',name)
            if a.exists() and b.exists():
                case_samples=directory/('cross-'+name);case_samples.mkdir(exist_ok=True)
                result['observed']['cross_'+name]=compare_traces(a,b,case_samples)
        keys=list(cross)
        result['first_nonidentical_boundary']=next((k for k in keys if not cross[k]['exact']),None)
        result['first_failing_boundary']=next((k for k in keys if not cross[k]['passed']),None)
        jumps=[dict(boundary=b,normalized_mse_increase=(cross[b]['normalized_mse'] or 0)-(cross[a]['normalized_mse'] or 0)) for a,b in zip(keys,keys[1:])]
        result['largest_error_increases']=sorted(jumps,key=lambda j:-j['normalized_mse_increase'])[:10]
        failure=result['first_failing_boundary']
        if failure:
            layer=int(failure[1:3]) if failure!='embedding' else None
            # A residual input discrepancy originates upstream; replay its preceding mixer.
            operation='norm' if failure.endswith('mixer_input') else 'mixer'
            if layer is not None and failure.endswith('resid_pre'): layer=max(0,layer-1)
            if layer is not None:
                key=f'L{layer:02d}_resid_pre' if operation=='norm' else f'L{layer:02d}_mixer_input'
                result['replay']=dict(layer=layer,operation=operation,case=primary,path=str(left.resolve()),key=key,
                    output_key=f'L{layer:02d}_mixer_input' if operation=='norm' else f'L{layer:02d}_mixer_out')
            result['unresolved'].append('A cross-backend boundary fails strict gates; the originating kernel is not yet established.')
        for backend_name in ('transformers','vllm'):
            path=args.output/(backend_name+'-replay')/(primary+'.safetensors')
            if path.exists():
                samples=directory/(backend_name+'-replay');samples.mkdir(exist_ok=True)
                result['observed'][backend_name+'_replay_vs_reference']=compare_traces(left,path,samples)
        if all((args.output/(b+'-replay')/(primary+'.safetensors')).exists() for b in ('transformers','vllm')):
            samples=directory/'common-input-replay';samples.mkdir(exist_ok=True)
            replay=compare_traces(args.output/'transformers-replay'/(primary+'.safetensors'),args.output/'vllm-replay'/(primary+'.safetensors'),samples)
            result['observed']['common_input_replay']=replay
            if result['replay']:
                output_key=result['replay']['output_key']
                input_key=f"L{result['replay']['layer']:02d}_replay_input"
                if input_key not in replay or not replay[input_key]['exact']:
                    raise ValueError('Operator replay inputs are not exactly identical')
                result['replay_result']=dict(boundary=output_key,agreement=replay[output_key],common_input_exact=True)
                result['supported_inference'].append('Common-input replay '+('still fails' if not replay[output_key]['passed'] else 'passes')+' strict gates at the selected operation; see native-input controls before attributing the cause.')
    else: result['unresolved'].append('Both complete anchor traces are required for cross-backend analysis.')
    result['unresolved'].append('Diagnostic timings include synchronization and must not be used as extraction speedups.')
    result['analysis_seconds']=time.perf_counter()-analysis_started
    write_json(args.output/'analysis.json',result)
    lines=['# Numerical divergence investigation','','The investigation tests where activation values first diverge while retaining the original precision thresholds.','',
        '## Observed','',f"First nonidentical boundary: `{result.get('first_nonidentical_boundary')}`. First failing boundary: `{result.get('first_failing_boundary')}`.",'',
        '| Boundary | Mean cosine | Minimum cosine | Normalised MSE | Strict gates |','|---|---:|---:|---:|---|']
    for k,v in result['observed'].get('cross_backend',{}).items():
        if not v['passed'] or k=='embedding': lines.append(f"| {k} | {v['mean_cosine']:.8f} | {v['minimum_cosine']:.8f} | {v['normalized_mse']} | {'pass' if v['passed'] else 'fail'} |")
    for label,key in (('Supported inference','supported_inference'),('Unresolved','unresolved')):
        lines+=['',f'## {label}','']+[f'- {s}' for s in result[key]]
    (args.output/'investigation.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:result.get(k) for k in ('first_nonidentical_boundary','first_failing_boundary','replay')}),flush=True)


def schedule(args):
    probes=[json.loads(p.read_text()) for p in args.output.glob('*-pilot/result.json')]
    records=[r for r in probes if r['status']=='complete']
    if {r['backend'] for r in records}!={'transformers','vllm'}: raise ValueError('Successful pilot receipts for both backends required')
    startup=sum(r['startup_seconds'] for r in records)
    extraction=sum(sum(c['diagnostic_wall_seconds'] for c in r['cases'].values()) for r in records)
    # Per backend: 1+8+64+8+64+4+short = <150 anchor-equivalent forwards.
    # Scale single-prompt probe by input tokens, conservatively without batch speedup.
    estimate=3*startup+150*extraction
    minutes=math.ceil((estimate*1.25+120)/300)*5
    write_json(args.output/'schedule.json',dict(measured_startup_seconds=startup,measured_probe_seconds=extraction,
        estimated_seconds=estimate,requested_minutes=min(minutes,60),requires_continuation=minutes>60,
        margin_fraction=.25,extra_seconds=120,failed_probes=[r.get('error') for r in probes if r['status']!='complete']))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='stage',required=True)
    for name in ('inspect','backend','analyse','schedule','moe-replay'):
        p=sub.add_parser(name);p.add_argument('--output',type=Path,required=True)
        if name in ('inspect','backend'):
            p.add_argument('--work',type=Path,required=True);p.add_argument('--reference',type=Path,required=True)
        if name=='backend':
            p.add_argument('--backend',choices=('transformers','vllm'),required=True)
            p.add_argument('--label',help='Separate receipt directory for an explicit runtime attempt')
            p.add_argument('--pilot',action='store_true');p.add_argument('--no-chunked-prefill',action='store_true')
            p.add_argument('--cases',nargs='+',help='Run only named cases; preserves full requested batch sizes')
            p.add_argument('--integrity',choices=('package','transformers','sibling_files','sibling_fills'))
            p.add_argument('--replay-selection',type=Path)
        if name=='moe-replay':
            p.add_argument('--work',type=Path,required=True)
            p.add_argument('--transformers-trace',type=Path,required=True)
            p.add_argument('--vllm-trace',type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(4)
    if args.stage=='moe-replay': return asyncio.run(moe_replay(args))
    return {'inspect':inspect_inputs,'backend':backend,'analyse':analyse,'schedule':schedule}[args.stage](args)


if __name__=='__main__':
    result=main()
    raise SystemExit(result if isinstance(result,int) else 0)
