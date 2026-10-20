"""Trace comparison and reports for the divergence investigation; CPU only."""
from __future__ import annotations

import json
import time

import torch
from safetensors.torch import load_file

from .common import versions, write_json
from .stats import TokenDiagnostics


def identities(tokens,sequence='anchor'):
    return [dict(row_id=f'{sequence}:{i:06d}',sequence_id=sequence,position=i,token_id=int(token))
            for i,token in enumerate(tokens.tolist())]


def boundary_order(key):
    if key=='embedding': return (-1,0)
    prefix,_,kind=key.partition('_')
    return (int(prefix[1:]),{'resid_pre':0,'mixer_input':1,'attn_prelinear':2,'mixer_out':3,'resid_post':4}.get(kind,99))


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


def write_moe_replay_report(output_dir,tensors,worker,engine_config,source,output):
    """Summarise a layer-1 MoE replay as layer1-moe-replay.json and .md."""
    summary={
        'status':'complete','worker':worker,'engine_configuration':engine_config,
        'source':source,
        'software':versions(),
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
    write_json(output_dir/'layer1-moe-replay.json',summary)
    lines=['# Layer 1 MoE anchor replay','','This replay separates the effect of the saved normalized input from the expert implementation. It uses the loaded vLLM BF16 weights for both the transparent PyTorch equations and the fused vLLM operation; the original Transformers and vLLM traces remain the native-output checks.','',
        '| Comparison | Mean cosine | Minimum cosine | Normalized MSE | Rows ≥ 0.990 |','|---|---:|---:|---:|---:|']
    for label,key in [('Saved cross-backend','saved_cross_backend'),('Manual replay vs saved Transformers','manual_replay_vs_saved_transformers'),('Native replay vs saved vLLM','native_replay_vs_saved_vllm'),('Route-matched manual vs native on Transformers input','route_matched_manual_vs_native_on_transformers_input'),('Route-matched manual vs native on vLLM input','route_matched_manual_vs_native_on_vllm_input')]:
        item=summary[key];lines.append(f"| {label} | {item['mean_cosine']:.8f} | {item['minimum_cosine']:.8f} | {item['normalized_mse']:.8g} | {item['rows_cosine_at_least_0_990']}/{item['rows']} |")
    lines+=['',f"The anchor-specific 99%/0.990 rule {'passes' if summary['saved_cross_backend']['passes_anchor_99_percent_rule'] else 'fails'} on the saved cross-backend outputs. The original strict mean, minimum and relative-L2 gates remain reported above.",'',f"TF32 router control changed expert sets for {summary['tf32_router_different_expert_rows']}/{summary['tf32_router_rows']} rows; maximum logit difference was {summary['tf32_router_maximum_logit_absolute_difference']:.8g}.",'','The decomposition is diagnostic only. It does not claim numerical equivalence or a fix. See the JSON receipt for software/configuration, effect norms, and limitations.']
    (output_dir/'layer1-moe-replay.md').write_text('\n'.join(lines)+'\n')
    return summary
