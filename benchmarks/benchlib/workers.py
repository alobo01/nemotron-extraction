"""vLLM worker extensions used only by the diagnostic benchmarks.

These synchronize for verification, so their timings are not extraction benchmarks.
"""
from __future__ import annotations

from pathlib import Path
import time

import torch
from safetensors.torch import load_file, save_file

from nemotron_extraction.hooks import install, layers_of, representative_sites
from nemotron_extraction.transport import SLOTS, atomic_path
from nemotron_extraction.worker import ActivationHookExtension, external_request_id

from .common import tensor_digest
from .moe import decompose_four_outputs, manual_nemotron_moe, manual_sigmoid_router, routing_report
from .sibling import sibling_namespace
from .stats import same_expert_sets
from .trace import TraceHooks, runtime_config

# Anchor rows whose routed experts are fingerprinted in the MoE replay report.
ANCHOR_POSITIONS = (571, 1467, 2005, 2021)


class DiagnosticWorker:
    """Anchor tracing, same-forward sibling comparison and MoE replay inside a vLLM worker."""
    def diag_install(self, reference):
        self.diag_trace = TraceHooks(self.model_runner.model, worker=True)
        self.diag_case = None
        self.diag_sibling = None
        self.diag_package_handles = []
        self.diag_handles = [self.model_runner.model.register_forward_pre_hook(self._diag_inputs,with_kwargs=True)]
        self.diag_reference = reference
        return runtime_config(self.model_runner.model,self.model_runner)

    def diag_begin(self, case, request_ids, token_ids, anchor, path, integrity=False, replay=None):
        if self.diag_case is not None:
            raise RuntimeError('Previous diagnostic case not drained')
        self.diag_trace.reset()
        self.diag_trace.replay = None if replay is None else dict(layer=replay['layer'],
            input=load_file(replay['path'])[replay['key']],operation=replay.get('operation','mixer'))
        if replay is not None and self.diag_trace.detail_layer is None:
            self.diag_trace.details(replay['layer'])
        self.diag_case = dict(name=case,requests=dict(zip(request_ids,token_ids)),anchor=request_ids[anchor],
            path=path,scheduler=[],covered=set(),integrity=integrity,equality=[],snapshots=[],steps=[])
        self.diag_package,self.diag_package_handles = {},[]
        self.diag_sibling = None
        if integrity:
            sites = representative_sites(self.model_runner.model)
            self.diag_package_handles,_ = install(self.model_runner.model,sites,self.diag_package,worker=True)
            source = sibling_namespace(self.diag_reference)
            sibling = source['ActivationHookExtension']()
            sibling.model_runner = self.model_runner
            sibling.sae_install_hooks([(s.key,s.kind,s.layer) for s in sites],None)
            self.diag_sibling = sibling
            self.diag_case['mode'] = integrity
            if integrity == 'sibling_files':
                sibling.sae_out_dir = Path(path).parent/(case+'-raw')
                sibling.sae_out_dir.mkdir(parents=True,exist_ok=True)
            # Compare before either end-of-step handler clears GPU captures.
            self.diag_package_handles.append(self.model_runner.model.register_forward_hook(self._diag_equal,prepend=True))
        return True

    def _diag_inputs(self,module,args,kwargs):
        case = self.diag_case
        if case is None:
            raise RuntimeError('Forward without diagnostic case')
        batch = self.model_runner.input_batch
        ids = list(batch.req_ids)
        starts = self.model_runner.query_start_loc.cpu[:len(ids)+1].tolist()
        self.diag_trace.span = None
        for i,internal in enumerate(ids):
            rid = external_request_id(internal, case['requests'])
            expected = case['requests'][rid]
            computed = int(batch.num_computed_tokens_cpu[i])
            start,stop = int(starts[i]),int(starts[i+1])
            case['scheduler'].append(dict(request_id=rid,internal_request_id=internal,start=start,stop=stop,
                computed_tokens=computed,prompt_length=len(expected),phase='decode' if computed>=len(expected) else 'prefill'))
            if computed>=len(expected):
                continue
            if computed or stop-start!=len(expected):
                raise RuntimeError('Partial prefill observed; diagnostic baseline requires whole prompts')
            if rid in case['covered']:
                raise RuntimeError('Duplicate prompt capture')
            if batch.token_ids_cpu[i,:len(expected)].tolist()!=expected:
                raise RuntimeError('Scheduler token IDs differ')
            case['covered'].add(rid)
            if rid==case['anchor']:
                self.diag_trace.span=(start,stop)

    def _diag_equal(self,module,args,output):
        case=self.diag_case
        if not case['integrity'] or not self.diag_package:
            return
        sibling=self.diag_sibling.sae_captured
        if self.diag_package.keys()!=sibling.keys():
            raise RuntimeError('Same-forward site sets differ')
        for key,left in self.diag_package.items():
            right=sibling[key]
            exact=left.shape==right.shape and left.dtype==right.dtype and torch.equal(left,right)
            case['equality'].append(dict(site=key,exact=exact,shape=list(left.shape),dtype=str(left.dtype)))
            if not exact:
                raise RuntimeError('Same-forward callbacks differ: '+key)
        # Synchronous oracle is diagnostic only. It precedes original transport.
        snapshot={k:t.detach().clone().cpu() for k,t in self.diag_package.items()}
        snapshot.update({k:t.detach().clone().cpu() for k,t in self.diag_sibling.sae_inputs.items()})
        case['snapshots'].append(snapshot)
        self.diag_package.clear()

    def diag_finish(self):
        case=self.diag_case
        if set(case['requests'])!=case['covered']:
            raise RuntimeError('Missing prompt spans')
        path=Path(case['path']);path.parent.mkdir(parents=True,exist_ok=True)
        tensors={k:v.cpu().contiguous() for k,v in self.diag_trace.values.items()}
        if not tensors or 'embedding' not in tensors:
            raise RuntimeError('Missing anchor trace')
        tensors['token_ids']=torch.tensor(case['requests'][case['anchor']],dtype=torch.long)
        if not all(bool(torch.isfinite(v).all()) for v in tensors.values()):
            raise ValueError('Nonfinite trace')
        with atomic_path(path) as partial_path:
            save_file(tensors,str(partial_path))
        for key,value in load_file(str(path)).items():
            if not torch.equal(value,tensors[key]):
                raise RuntimeError('Trace persistence mismatch '+key)
        report=dict(name=case['name'],scheduler=case['scheduler'],same_forward=case['equality'],
            topology=self.diag_trace.topology,path=str(path),trace_persistence_exact=True,
            operator_observability=self.diag_trace.detail_observability)
        # Capture the real selector's logits, IDs, and weights while the
        # worker still owns the loaded layer. This avoids inferring routes
        # from an independently reimplemented top-k rule.
        if self.diag_trace.detail_layer is not None:
            layer_index=self.diag_trace.detail_layer
            prefix=f'L{layer_index:02d}_detail_'
            logits=self.diag_trace.values.get(prefix+'gate_0')
            selection=[(key,value) for key,value in self.diag_trace.values.items()
                       if key.startswith(prefix+'actual_router_selection_')]
            ids=next((value for _,value in selection if value.dtype in
                      (torch.int8,torch.int16,torch.int32,torch.int64,torch.uint8)),None)
            weights=next((value for _,value in selection if value.is_floating_point()),None)
            _,layers=layers_of(self.model_runner.model)
            correction=getattr(getattr(layers[layer_index].mixer,'gate',None),
                               'e_score_correction_bias',None)
            if logits is not None and ids is not None and weights is not None and correction is not None:
                routes=routing_report(logits,correction,ids)
                margins=routes['global_sixth_seventh_margin']
                agreement=routes['unconstrained_set_agreement']
                report['routing']={
                    'rows':int(ids.shape[0]),'top_k':int(ids.shape[1]),
                    'ids_dtype':str(ids.dtype),'weights_dtype':str(weights.dtype),
                    'logits_dtype':str(logits.dtype),
                    'global_kth_to_best_excluded_margin_min':float(margins.min()),
                    'global_kth_to_best_excluded_margin_mean':float(margins.mean()),
                    'unconstrained_global_topk_set_agreement':int(agreement.sum()),
                    'unconstrained_global_topk_rows':int(agreement.numel()),
                    'observed_expert_ids':ids.cpu().tolist(),
                    'observed_routing_weights':weights.cpu().tolist(),
                    'global_kth_to_best_excluded_margin':margins.cpu().tolist(),
                }
                report['routing']['note']=(
                    'Global score margins do not encode grouped eligibility; '
                    'observed selected IDs and weights are the backend selector output.')
        if self.diag_sibling is not None:
            source=sibling_namespace(self.diag_reference)
            rows=torch.tensor(list(case['requests'].values()),dtype=torch.long)
            keys=list(case['snapshots'][0].keys()-{'input_ids','positions'})
            oracle=source['assemble_rows'](case['snapshots'],rows,keys)
            if case['mode']=='sibling_files':
                saved=[load_file(p) for p in self.diag_sibling.sae_flush()]
                restored=source['assemble_rows'](saved,rows,keys)
            else:
                directory=path.parent/(case['name']+'-fill')
                self.diag_sibling.sae_save(rows.tolist(),str(directory),4)
                self.diag_sibling.sae_flush_saves()
                if not torch.equal(load_file(str(directory/'token_ids.safetensors'))['token_ids'],rows):
                    raise RuntimeError('Sibling persisted token order differs')
                restored={k:load_file(str(directory/(k+'.safetensors')))['activations'] for k in keys}
            report['sibling_transport_exact']=all(torch.equal(oracle[k],restored[k]) for k in keys)
            if not report['sibling_transport_exact']:
                raise RuntimeError('Sibling transfer/persistence mismatch')
            # Retain an inspectable oracle; package transport has its own same-forward check.
            oracle['token_ids']=rows
            save_file(oracle,str(path.with_suffix('.oracle.safetensors')))
            self.diag_sibling.sae_remove_hooks()
        for handle in self.diag_package_handles:
            handle.remove()
        self.diag_trace.reset()
        self.diag_case=None
        return report

    def diag_close(self):
        if self.diag_sibling is not None and self.diag_case is not None:
            self.diag_sibling.sae_remove_hooks()
        for handle in self.diag_package_handles:
            handle.remove()
        for handle in self.diag_handles:
            handle.remove()
        self.diag_trace.close()
        self.diag_case=None

    def diag_moe_replay(self, transformer_trace, vllm_trace, output_path, layer_index=1):
        """Compare transparent PyTorch expert equations with this worker's fused MoE."""
        base,layers=layers_of(self.model_runner.model)
        mixer=layers[layer_index].mixer
        params=dict(mixer.named_parameters())
        w13_name=next((n for n in params if n.endswith('w13_weight')),None)
        w2_name=next((n for n in params if n.endswith('w2_weight')),None)
        if w13_name is None or w2_name is None:
            raise RuntimeError('Cannot resolve loaded routed expert weights')
        w13,w2=params[w13_name],params[w2_name]
        if (w13.ndim!=3 or w2.ndim!=3 or w13.shape[0]!=w2.shape[0]
            or w13.shape[1]!=w2.shape[2] or w13.shape[2]!=w2.shape[1]):
            raise RuntimeError(f'Unsupported non-gated expert weight layout: {tuple(w13.shape)} / {tuple(w2.shape)}')
        if w13.dtype!=torch.bfloat16 or w2.dtype!=torch.bfloat16:
            raise RuntimeError(f'Expected BF16 expert weights, got {w13.dtype} / {w2.dtype}')
        router=getattr(mixer.experts,'router',None)
        select=getattr(router,'select_experts',None)
        if not callable(select):
            raise RuntimeError('Cannot intercept this vLLM router selector')
        shared=getattr(mixer,'shared_experts',None)
        def projection(name):
            module=getattr(shared,name,None) if shared is not None else None
            return (getattr(module,'weight',None),getattr(module,'bias',None)) if module is not None else (None,None)
        shared_up,shared_up_bias=projection('up_proj')
        shared_down,shared_down_bias=projection('down_proj')
        if (shared_up is None)!=(shared_down is None):
            raise RuntimeError('Could not resolve both shared expert projections')

        h=load_file(str(transformer_trace));v=load_file(str(vllm_trace))
        input_key=f'L{layer_index:02d}_mixer_input';output_key=f'L{layer_index:02d}_mixer_out'
        if any(key not in trace for trace in (h,v) for key in (input_key,output_key)):
            raise RuntimeError('Saved traces lack the selected layer input/output')
        if 'token_ids' not in h or 'token_ids' not in v or not torch.equal(h['token_ids'],v['token_ids']):
            raise RuntimeError('Saved traces do not have identical anchor token IDs')
        xh=h[input_key].to(device=w13.device,dtype=w13.dtype)
        xv=v[input_key].to(device=w13.device,dtype=w13.dtype)
        if xh.ndim!=2 or xv.shape!=xh.shape or xh.shape[-1]!=w13.shape[2]:
            raise RuntimeError('Saved normalized inputs do not match loaded expert dimensions')

        model_config=getattr(getattr(getattr(self.model_runner,'vllm_config',None),'model_config',None),
                             'hf_config',None)
        if model_config is None:
            raise RuntimeError('Cannot verify the pinned model routing configuration')
        if int(getattr(model_config,'n_group',0))!=1 or int(getattr(model_config,'topk_group',0))!=1:
            raise RuntimeError('Manual router currently requires the pinned one-group Nemotron router')
        gate_weight=getattr(mixer.gate,'weight',None)
        if gate_weight is None:
            raise RuntimeError('Cannot resolve the loaded FP32 router matrix')
        correction=mixer.gate.e_score_correction_bias
        top_k=int(getattr(model_config,'num_experts_per_tok',0))
        if top_k<=0: raise RuntimeError('Cannot resolve the checkpoint top-k value')
        expected=(int(getattr(model_config,'n_routed_experts',w13.shape[0])),
                  int(getattr(model_config,'moe_intermediate_size',w13.shape[1])),
                  int(getattr(model_config,'hidden_size',w13.shape[2])))
        observed=(int(w13.shape[0]),int(w13.shape[1]),int(w13.shape[2]))
        if observed!=expected:
            raise RuntimeError(f'Loaded expert weight dimensions {observed} disagree with checkpoint config {expected}')
        routed_scale=float(getattr(mixer,'routed_scaling_factor',1.0))

        def manual_route(x,enabled=False):
            old_tf32=torch.backends.cuda.matmul.allow_tf32
            old_precision=torch.get_float32_matmul_precision()
            try:
                torch.backends.cuda.matmul.allow_tf32=enabled
                torch.set_float32_matmul_precision('high' if enabled else 'highest')
                return manual_sigmoid_router(x,gate_weight,correction,top_k,
                    gate_bias=getattr(mixer.gate,'bias',None),routed_scale=1.0)
            finally:
                torch.backends.cuda.matmul.allow_tf32=old_tf32
                torch.set_float32_matmul_precision(old_precision)

        def native(x):
            observed={}
            def capture_selection(*args,**kwargs):
                result=select(*args,**kwargs)
                if 'router_logits' in kwargs: observed['logits']=kwargs['router_logits'].detach()
                if isinstance(result,(tuple,list)) and len(result)>=2:
                    observed['weights'],observed['ids']=result[0].detach(),result[1].detach()
                return result
            router.select_experts=capture_selection
            try:
                with torch.inference_mode(): output=mixer(x)
            finally:
                router.select_experts=select
            if 'ids' not in observed:
                raise RuntimeError('Selector did not return observed weights and IDs')
            return output,observed

        def manual(x,routes):
            return manual_nemotron_moe(x,routes['ids'],routes['weights'],w13,w2,
                shared_up_weight=shared_up,shared_down_weight=shared_down,
                shared_up_bias=shared_up_bias,shared_down_bias=shared_down_bias,
                routed_scale=routed_scale,output_dtype=x.dtype)

        with torch.inference_mode():
            native_h,routes_h=native(xh);native_v,routes_v=native(xv)
            manual_router_h=manual_route(xh);manual_router_v=manual_route(xv)
            manual_h=manual(xh,manual_router_h);manual_v=manual(xv,manual_router_v)
            matched_h=manual(xh,routes_h);matched_v=manual(xv,routes_v)
            values=decompose_four_outputs(manual_h['output'],manual_v['output'],native_h,native_v)
            tf32_results={}
            for enabled in (False,True):
                tf32_results[str(enabled)]=manual_route(xh,enabled=enabled)
        tensors={
            'token_ids':h['token_ids'],
            'saved_transformers_output':h[output_key],'saved_vllm_output':v[output_key],
            'manual_on_transformers_input':manual_h['output'].cpu(),
            'native_on_transformers_input':native_h.cpu(),
            'manual_on_vllm_input':manual_v['output'].cpu(),
            'native_on_vllm_input':native_v.cpu(),
            'route_matched_manual_on_transformers_input':matched_h['output'].cpu(),
            'route_matched_manual_on_vllm_input':matched_v['output'].cpu(),
            'input_effect':values['input_effect'].cpu(),
            'implementation_effect_on_candidate_input':values['implementation_effect_on_candidate_input'].cpu(),
            'implementation_effect_on_reference_input':values['implementation_effect_on_reference_input'].cpu(),
            'interaction_effect':values['interaction_effect'].cpu(),
            'route_ids_transformers_input':routes_h['ids'].cpu(),
            'route_weights_transformers_input':routes_h['weights'].cpu(),
            'route_ids_vllm_input':routes_v['ids'].cpu(),
            'route_weights_vllm_input':routes_v['weights'].cpu(),
            'manual_route_ids_transformers_input':manual_router_h['ids'].cpu(),
            'manual_route_weights_transformers_input':manual_router_h['weights'].cpu(),
            'manual_route_ids_vllm_input':manual_router_v['ids'].cpu(),
            'manual_route_weights_vllm_input':manual_router_v['weights'].cpu(),
            'router_logits_vllm_transformers_input':routes_h['logits'].cpu(),
            'router_logits_vllm_vllm_input':routes_v['logits'].cpu(),
            'vllm_kth_to_best_excluded_margin_transformers_input':routing_report(
                routes_h['logits'],correction,routes_h['ids'])['global_sixth_seventh_margin'].cpu(),
            'vllm_kth_to_best_excluded_margin_vllm_input':routing_report(
                routes_v['logits'],correction,routes_v['ids'])['global_sixth_seventh_margin'].cpu(),
            'router_logits_tf32_off':tf32_results['False']['logits'].cpu(),
            'router_logits_tf32_on':tf32_results['True']['logits'].cpu(),
            'route_ids_tf32_off':tf32_results['False']['ids'].cpu(),
            'route_ids_tf32_on':tf32_results['True']['ids'].cpu(),
            'route_weights_tf32_off':tf32_results['False']['weights'].cpu(),
            'route_weights_tf32_on':tf32_results['True']['weights'].cpu(),
            'routed_manual_transformers_input':manual_h['routed_cast'].cpu(),
            'routed_manual_vllm_input':manual_v['routed_cast'].cpu(),
            'routed_route_matched_transformers_input':matched_h['routed_cast'].cpu(),
            'routed_route_matched_vllm_input':matched_v['routed_cast'].cpu(),
        }
        path=Path(output_path)
        with atomic_path(path) as partial:
            save_file(tensors,str(partial),metadata={
                'layer':str(layer_index),'up_parameter':w13_name,'down_parameter':w2_name,
                'up_dtype':str(w13.dtype),'down_dtype':str(w2.dtype),
                'up_shape':str(tuple(w13.shape)),'down_shape':str(tuple(w2.shape)),
                'torch_version':str(torch.__version__),'cuda_version':str(torch.version.cuda),
                'tf32_matmul':str(torch.backends.cuda.matmul.allow_tf32),
            })
        same_rows=same_expert_sets(routes_h['ids'],routes_v['ids'])
        tf32_same=same_expert_sets(tf32_results['False']['ids'],tf32_results['True']['ids'])
        logits_delta=(tf32_results['False']['logits'].float()-tf32_results['True']['logits'].float()).abs()
        router_match_h=same_expert_sets(routes_h['ids'],manual_router_h['ids'])
        router_match_v=same_expert_sets(routes_v['ids'],manual_router_v['ids'])
        def aligned_weight_delta(native_routes,manual_routes):
            native_order=native_routes['ids'].long().argsort(dim=-1)
            manual_order=manual_routes['ids'].long().argsort(dim=-1)
            native_ids=native_routes['ids'].long().gather(1,native_order)
            manual_ids=manual_routes['ids'].long().gather(1,manual_order)
            equal=native_ids==manual_ids
            native_weights=native_routes['weights'].float().gather(1,native_order)
            manual_weights=manual_routes['normalized_weights'].gather(1,manual_order)
            return (native_weights-manual_weights).abs()[equal]
        weight_deltas=[aligned_weight_delta(routes_h,manual_router_h),
                       aligned_weight_delta(routes_v,manual_router_v)]
        weight_deltas=[d for d in weight_deltas if d.numel()]
        selected_for_fingerprint=set()
        for position in ANCHOR_POSITIONS:
            if position<len(xh):
                selected_for_fingerprint.update(routes_h['ids'][position].long().tolist())
                selected_for_fingerprint.update(routes_v['ids'][position].long().tolist())
                selected_for_fingerprint.update(manual_router_h['ids'][position].long().tolist())
        router_fingerprints={
            'gate_weight_sha256':tensor_digest(gate_weight),
            'correction_bias_sha256':tensor_digest(correction),
            'correction_bias_values':correction.detach().float().cpu().tolist(),
            'experts':{
                str(expert):{'up_sha256':tensor_digest(w13[expert]),'down_sha256':tensor_digest(w2[expert]),
                    'up_dtype':str(w13.dtype),'down_dtype':str(w2.dtype),
                    'up_shape':list(w13[expert].shape),'down_shape':list(w2[expert].shape)}
                for expert in sorted(selected_for_fingerprint)
            },
            'expert_ids_fingerprinted_at_anchor_positions':list(ANCHOR_POSITIONS),
        }
        return {
            'status':'complete','layer':layer_index,'tokens':int(xh.shape[0]),'hidden_width':int(xh.shape[1]),
            'module_path':f'{base}.{layer_index}.mixer',
            'top_k':int(routes_h['ids'].shape[1]),'expert_weight_names':{'up':w13_name,'down':w2_name},
            'expert_weight_shapes':{'up':list(w13.shape),'down':list(w2.shape)},
            'expert_weight_dtypes':{'up':str(w13.dtype),'down':str(w2.dtype)},
            'loaded_weight_fingerprints':router_fingerprints,
            'checkpoint_config_dimensions':{
                'hidden_size':expected[2],'routed_experts':expected[0],'intermediate_size':expected[1],
                'top_k':top_k,'n_group':1,'topk_group':1,
            },
            'shared_expert_present':shared_up is not None,
            'routed_scaling_factor':routed_scale,
            'expert_set_agreement_rows':int(same_rows.sum()),'expert_set_total_rows':int(same_rows.numel()),
            'manual_router_vs_vllm':{
                'transformers_input_matching_set_rows':int(router_match_h.sum()),
                'vllm_input_matching_set_rows':int(router_match_v.sum()),
                'rows':int(xh.shape[0]),
                'max_weight_difference_on_matching_selected_experts':float(max(d.max() for d in weight_deltas)) if weight_deltas else None,
                'vllm_logits_vs_manual_fp32_max_abs_transformers_input':float((routes_h['logits'].float()-manual_router_h['logits']).abs().max()),
                'vllm_logits_vs_manual_fp32_max_abs_vllm_input':float((routes_v['logits'].float()-manual_router_v['logits']).abs().max()),
                'correction_bias_applies_to_selection_only':True,
            },
            'selected_expert_margin':{
                'meaning':'minimum selected corrected sigmoid score minus maximum excluded corrected sigmoid score; global one-group ranking',
                'transformers_input_minimum':float(tensors['vllm_kth_to_best_excluded_margin_transformers_input'].min()),
                'vllm_input_minimum':float(tensors['vllm_kth_to_best_excluded_margin_vllm_input'].min()),
                'positions_of_interest':{
                    str(position):{
                        'transformers_input':float(tensors['vllm_kth_to_best_excluded_margin_transformers_input'][position]),
                        'vllm_input':float(tensors['vllm_kth_to_best_excluded_margin_vllm_input'][position]),
                    } for position in ANCHOR_POSITIONS if position<len(xh)
                },
            },
            'tf32_router_control':{
                'different_expert_set_rows':int((~tf32_same).sum()),'rows':int(tf32_same.numel()),
                'maximum_logit_absolute_difference':float(logits_delta.max()),
                'mean_logit_absolute_difference':float(logits_delta.mean()),
                'tested_input':'Transformers anchor normalized input',
                'disabled':{'allow_tf32':False,'float32_matmul_precision':'highest'},
                'permitted':{'allow_tf32':True,'float32_matmul_precision':'high'},
                'caveat':'PyTorch high precision may select TF32 or another accelerated FP32 algorithm; the effective kernel must be confirmed from this runtime.',
            },
            'output_path':str(path),
            'note':'Manual equations use observed vLLM router outputs; native fused outputs are on the same loaded weights/runtime.'
        }


class IntegrityWorker(ActivationHookExtension):
    """Verify production asynchronous copies against same-forward GPU snapshots."""
    def nh_install(self,sites,sink_dir,diagnostics=True,publication_threads=1):
        result=super().nh_install(sites,sink_dir,diagnostics,publication_threads)
        self.nh_handles.append(self.model_runner.model.register_forward_hook(self._oracle,prepend=True))
        return result

    def _oracle(self,module,args,output):
        wave=self.nh_active
        if not self.nh_spans:
            return
        oracle=wave.setdefault('diagnostic_oracle',{})
        length=wave['tokens'].shape[1]
        for key,source in self.nh_captured.items():
            host=oracle.setdefault(key,torch.empty((wave['tokens'].numel(),source.shape[-1]),dtype=source.dtype))
            for index,start,stop in self.nh_spans:
                host[index*length:(index+1)*length].copy_(source[start:stop].detach().clone().cpu())
        wave['diagnostic_slot']=wave['slot']
        wave['diagnostic_occupied_slots']=SLOTS-len(self.nh_pool.free)

    @staticmethod
    def _nh_publish(wave):
        time.sleep(.5)  # hold this slot so the next wave fills the other buffer
        result=ActivationHookExtension._nh_publish(wave)
        saved=load_file(result['path'])
        oracle=wave['diagnostic_oracle']
        exact=all(saved[k].dtype==v.dtype and saved[k].shape==v.shape and torch.equal(saved[k],v)
                  for k,v in oracle.items()) and torch.equal(saved['token_ids'],wave['tokens'])
        if not exact:
            raise RuntimeError('Package asynchronous transfer/persistence mismatch')
        result.update(diagnostic_transport_exact=exact,diagnostic_slot=wave['diagnostic_slot'],
                      diagnostic_occupied_slots=wave['diagnostic_occupied_slots'])
        return result
