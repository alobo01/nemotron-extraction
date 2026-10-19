"""Opt-in numerical diagnostics; never imported by the extraction hot path.

Sibling callbacks retain their attribution through the supplied source snapshot.
This module deliberately synchronizes for verification, so its timings are not
production extraction benchmarks.
"""
from __future__ import annotations

import ast
from concurrent.futures import Future, ThreadPoolExecutor
from functools import partial
import hashlib
import inspect
import json
from pathlib import Path
import time
from collections.abc import Iterable

import torch
from torch import nn
from safetensors.torch import load_file, save_file

from .hooks import argument, install, layers_of, mixer_kind, projection_of
from .types import HookSite
from .worker import ActivationHookExtension


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class TokenDiagnostics:
    """Exact scalar distributions with at most 40 retained activation pairs.

    Row identity is supplied explicitly; it never assumes a fixed sequence length.
    Shards can be merged without changing ranking or aggregate statistics.
    """
    def __init__(self):
        self.rows = []
        self.samples = {}
        self.width = self.dtype = None

    def add(self, reference, candidate, identities):
        if reference.shape != candidate.shape or reference.dtype != candidate.dtype:
            raise ValueError("Activation shapes or dtypes differ")
        if reference.ndim != 2 or len(identities) != len(reference):
            raise ValueError("Expected one identity per activation row")
        if self.width is not None and (reference.shape[1] != self.width or str(reference.dtype) != self.dtype):
            raise ValueError("Incompatible diagnostic shards")
        self.width, self.dtype = reference.shape[1], str(reference.dtype)
        h, v = reference.detach().cpu().float(), candidate.detach().cpu().float()
        if not bool(torch.isfinite(h).all() and torch.isfinite(v).all()):
            raise ValueError("Nonfinite activations")
        hn, vn = h.norm(dim=-1), v.norm(dim=-1)
        cos = ((h*v).sum(-1) / (hn*vn).clamp_min(torch.finfo(torch.float32).tiny)).clamp(-1, 1)
        cos[(hn == 0) & (vn == 0)] = 1
        err = (v-h).double().square().sum(-1)
        ref = h.double().square().sum(-1)
        if len({r['row_id'] for r in identities}) != len(identities):
            raise ValueError("Duplicate row identity")
        existing = {r['row_id'] for r in self.rows}
        for i, identity in enumerate(identities):
            if identity['row_id'] in existing:
                raise ValueError("Duplicate row identity")
            self.rows.append(dict(identity, cosine=float(cos[i]), reference_norm=float(hn[i]),
                candidate_norm=float(vn[i]), error_squared=float(err[i]), reference_squared=float(ref[i]),
                maximum_absolute_error=float((v[i]-h[i]).abs().max()),
                relative_error=float(err[i].sqrt()/hn[i]) if hn[i] else (0. if not vn[i] else None),
                relative_error_undefined=bool(hn[i] == 0 and vn[i] != 0),
                exact=bool(torch.equal(reference[i], candidate[i]))))
        # Only candidates for either global top-20 need their full tensors kept.
        local = set(sorted(range(len(identities)),key=lambda i:(float(cos[i]),identities[i]['row_id']))[:20])
        local.update(sorted(range(len(identities)),key=lambda i:(-float(err[i]),identities[i]['row_id']))[:20])
        for i in local:
            self.samples[identities[i]['row_id']] = (reference[i].detach().cpu().clone(), candidate[i].detach().cpu().clone())
        self._prune()

    def _rank(self, key, reverse=False):
        return sorted(self.rows, key=lambda r: ((-r[key] if reverse else r[key]), r['row_id']))[:20]

    def _prune(self):
        keep = {r['row_id'] for r in self._rank('cosine') + self._rank('error_squared', True)}
        self.samples = {k: v for k, v in self.samples.items() if k in keep}

    def merge(self, other):
        if self.width is not None and (self.width, self.dtype) != (other.width, other.dtype):
            raise ValueError("Incompatible diagnostic shards")
        if {r['row_id'] for r in self.rows} & {r['row_id'] for r in other.rows}:
            raise ValueError("Duplicate row identity")
        self.width, self.dtype = other.width, other.dtype
        self.rows.extend(other.rows)
        self.samples.update(other.samples)
        self._prune()
        return self

    def result(self):
        if not self.rows:
            raise ValueError("No diagnostic rows")
        rows = sorted(self.rows, key=lambda r: r['row_id'])
        cos = torch.tensor([r['cosine'] for r in rows], dtype=torch.float64)
        norms = torch.tensor([r['reference_norm'] for r in rows], dtype=torch.float64)
        error2 = sum(r['error_squared'] for r in rows)
        reference2 = sum(r['reference_squared'] for r in rows)
        nmse = error2/reference2 if reference2 else (0. if not error2 else None)
        relative = nmse**.5 if nmse is not None else None
        ranked_error = self._rank('error_squared', True)
        count = max(1, int(.001*len(rows)))
        top_error = sorted((r['error_squared'] for r in rows), reverse=True)[:count]
        order = torch.argsort(norms, stable=True)
        deciles = []
        for number, indices in enumerate(torch.tensor_split(order, 10)):
            if indices.numel():
                c, n = cos[indices], norms[indices]
                deciles.append(dict(decile=number+1, rows=len(indices), reference_norm_min=float(n.min()),
                    reference_norm_max=float(n.max()), mean_cosine=float(c.mean()), minimum_cosine=float(c.min()),
                    first_percentile_cosine=float(c.quantile(.01))))
        def enriched(items):
            return [dict(r, error_energy_share=r['error_squared']/error2 if error2 else 0.) for r in items]
        mean, minimum = float(cos.mean()), float(cos.min())
        return dict(tokens=len(rows), width=self.width, dtype=self.dtype, mean_cosine=mean,
            cosine_std=float(cos.std(correction=0)), minimum_cosine=minimum,
            cosine_percentiles={str(p): float(cos.quantile(p/100)) for p in (0, .1, 1, 5, 50, 99, 100)},
            relative_l2=relative, normalized_mse=nmse, mse_normalization='shared_reference_rms_per_site',
            error_squared_sum=error2, reference_squared_sum=reference2,
            maximum_absolute_error=max(r['maximum_absolute_error'] for r in rows),
            exact=all(r['exact'] for r in rows), both_zero_rows=sum(r['reference_norm']==0 and r['candidate_norm']==0 for r in rows),
            one_zero_rows=sum((r['reference_norm']==0) != (r['candidate_norm']==0) for r in rows),
            high_relative_error_fraction=sum(r['relative_error_undefined'] or r['relative_error'] > .02 for r in rows)/len(rows),
            worst_point_one_percent_error_share=sum(top_error)/error2 if error2 else 0.,
            reference_norm_deciles=deciles, worst_cosine=enriched(self._rank('cosine')),
            worst_error=enriched(ranked_error),
            passed=mean >= .9999 and minimum >= .9990 and relative is not None and relative <= .02)

    def save_samples(self, path):
        ordered = sorted(self.samples)
        save_file({f'{i}_{side}': self.samples[key][j] for i,key in enumerate(ordered)
                   for j,side in enumerate(('reference','candidate'))}, str(path),
                  metadata={'row_ids': json.dumps(ordered)})


def sibling_namespace(directory):
    """Execute attributed callback definitions without training/queue imports."""
    scope = dict(torch=torch, nn=nn, partial=partial, Path=Path, json=json, time=time,
                 ThreadPoolExecutor=ThreadPoolExecutor, Future=Future, Iterable=Iterable, save_file=save_file)
    for filename in ('attention.py', 'worker.py'):
        path = Path(directory)/filename
        tree = ast.parse(path.read_text())
        nodes = [ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)]
        nodes += [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) or
                  isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in
                  ('MIXER_KINDS','RESIDUAL_KINDS') for t in n.targets)]
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(path), 'exec'), scope)
    return scope


def tensor_output(value):
    return value[0] if isinstance(value, tuple) else value


def runtime_config(model, runner=None):
    """Runtime values and source hashes, rather than inferred library defaults."""
    sources, modules = {}, {}
    for name, module in model.named_modules():
        path = inspect.getsourcefile(type(module))
        if path and path not in sources and Path(path).is_file():
            sources[path] = digest(path)
        interesting = {key: str(getattr(module,key)) for key in
            ('ssm_state_dtype','conv_state_dtype','mamba_ssm_cache_dtype','mamba_cache_dtype',
             'attn_backend','quant_method','scoring_func','top_k','routed_scaling_factor') if hasattr(module,key)}
        if interesting:
            modules[name] = dict(type=type(module).__name__, **interesting)
    config = getattr(model,'vllm_config',getattr(runner,'vllm_config',None))
    state_dtypes=None
    if config is not None and hasattr(model,'get_mamba_state_dtype_from_config'):
        state_dtypes=[str(value) for value in model.get_mamba_state_dtype_from_config(config)]
    return dict(model_class=type(model).__name__, sources=sources, modules=modules,
        weight_dtypes=sorted({str(p.dtype) for p in model.parameters()}),
        cache_config=str(getattr(config,'cache_config',None)),
        resolved_mamba_state_dtypes=state_dtypes,
        model_config=str(getattr(config,'model_config',None)),
        # vLLM's worker RPC serializer only accepts builtin strings here;
        # TorchVersion is a str subclass that it rejects by exact type.
        torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
        tf32_matmul=torch.backends.cuda.matmul.allow_tf32, tf32_cudnn=torch.backends.cudnn.allow_tf32,
        matmul_precision=torch.get_float32_matmul_precision())


class TraceHooks:
    """Capture anchor rows at explicit mathematical boundaries, before mutation."""
    def __init__(self, model, worker=False):
        self.model, self.worker = model, worker
        self.handles, self.values, self.topology = [], {}, {}
        self.span = None
        self.replay = None
        self.detail_layer = None
        self.detail_handles = []
        self.detail_observability = {}
        self.restore_methods = []
        base, layers = layers_of(model)
        for name,module in model.named_modules():
            if name in ('model.embed_tokens','backbone.embeddings','backbone.embed_tokens','embed_tokens','embeddings'):
                self.handles.append(module.register_forward_hook(partial(self.post, 'embedding', name)))
                break
        else:
            raise ValueError('Cannot resolve embedding module')
        for i, block in enumerate(layers):
            path = f'{base}.{i}'
            self.handles.append(block.register_forward_pre_hook(partial(self.block_pre, i, path), with_kwargs=True))
            self.handles.append(block.register_forward_hook(partial(self.block_post, i, path)))
            self.handles.append(block.mixer.register_forward_pre_hook(partial(self.mixer_pre, i, path+'.mixer'), with_kwargs=True))
            self.handles.append(block.mixer.register_forward_hook(partial(self.post, f'L{i:02d}_mixer_out', path+'.mixer')))
            self.handles.append(block.norm.register_forward_pre_hook(partial(self.norm_pre,i,path+'.norm'),with_kwargs=True))
            if mixer_kind(block.mixer) == 'attn_out':
                name, projection = projection_of(block.mixer)
                self.handles.append(projection.register_forward_pre_hook(
                    partial(self.input_hook, f'L{i:02d}_attn_prelinear', path+'.mixer.'+name), with_kwargs=True))

    def rows(self, value):
        if self.span is None:
            return None
        start, stop = self.span
        return value.reshape(-1,value.shape[-1])[start:stop]

    def keep(self, key, path, value):
        value = self.rows(value)
        if value is None:
            return
        self.keep_rows(key,path,value)

    def keep_rows(self,key,path,value):
        if key in self.values:
            raise RuntimeError('Anchor captured twice at '+key)
        self.values[key] = value.detach().clone()
        self.topology[key] = dict(module_path=path, width=value.shape[-1], dtype=str(value.dtype))

    def block_pre(self, layer, path, module, args, kwargs):
        x = argument(args,kwargs,1 if self.worker else 0,'hidden_states')
        if self.worker:
            r = argument(args,kwargs,2,'residual')
            self.keep(f'L{layer:02d}_raw_hidden',path,x)
            if r is not None:
                self.keep(f'L{layer:02d}_raw_residual',path,r)
            if self.span is not None:
                self.keep_rows(f'L{layer:02d}_resid_pre',path,self.rows(x) if r is None else self.rows(x)+self.rows(r))
        else:
            self.keep(f'L{layer:02d}_resid_pre',path,x)

    def block_post(self, layer, path, module, args, output):
        if self.worker:
            self.keep(f'L{layer:02d}_raw_out_hidden',path,output[0])
            self.keep(f'L{layer:02d}_raw_out_residual',path,output[1])
            if self.span is not None:
                self.keep_rows(f'L{layer:02d}_resid_post',path,self.rows(output[0])+self.rows(output[1]))
            return
        else:
            value = tensor_output(output)
        self.keep(f'L{layer:02d}_resid_post',path,value)

    def mixer_pre(self, layer, path, module, args, kwargs):
        # Attention mixers take positions before hidden_states in vLLM.
        index = 1 if self.worker and mixer_kind(module) == 'attn_out' else 0
        value = argument(args,kwargs,index,'hidden_states')
        self.keep(f'L{layer:02d}_mixer_input',path,value)
        if self.replay is not None and self.replay.get('operation','mixer') == 'mixer' and layer == self.replay['layer'] and self.span is not None:
            value = value.clone()
            replacement = self.replay['input'].to(device=value.device,dtype=value.dtype)
            start,stop = self.span
            value.reshape(-1,value.shape[-1])[start:stop].copy_(replacement)
            if 'hidden_states' in kwargs:
                kwargs = dict(kwargs,hidden_states=value)
            else:
                args = list(args)
                args[index] = value
                args = tuple(args)
            self.keep(f'L{layer:02d}_replay_input',path,value)
            return args,kwargs

    def norm_pre(self,layer,path,module,args,kwargs):
        if self.replay is None or self.replay.get('operation')!='norm' or layer!=self.replay['layer'] or self.span is None:
            return
        x=argument(args,kwargs,0,'x' if self.worker else 'hidden_states').clone()
        start,stop=self.span
        x.reshape(-1,x.shape[-1])[start:stop].copy_(self.replay['input'].to(x))
        args=list(args);kwargs=dict(kwargs)
        name='x' if self.worker else 'hidden_states'
        if name in kwargs: kwargs[name]=x
        else: args[0]=x
        if self.worker:
            r=kwargs.get('residual',args[1] if len(args)>1 else None)
            if r is not None:
                r=r.clone();r.reshape(-1,r.shape[-1])[start:stop].zero_()
                if 'residual' in kwargs: kwargs['residual']=r
                else: args[1]=r
        self.keep(f'L{layer:02d}_replay_input',path,x)
        return tuple(args),kwargs

    def input_hook(self, key, path, module, args, kwargs):
        self.keep(key,path,argument(args,kwargs,0,'hidden_states'))

    def details(self,layer):
        """Observe executed submodules; never claim fused internals were observed."""
        self.detail_layer=layer
        base,layers=layers_of(self.model)
        mixer=layers[layer].mixer
        wanted=('in_proj','out_proj','o_proj','q_proj','k_proj','v_proj','qkv_proj','conv1d','norm','gate','shared_experts','experts')
        self.detail_observability=dict(mixer_type=type(mixer).__name__,requested=list(wanted),observed=[],
            routing_ids='unobserved',state_policy='new request, position zero; no reused request cache')
        self.detail_observability['parameters']={name:dict(shape=list(p.shape),dtype=str(p.dtype),
            sha256=hashlib.sha256(p.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest())
            for name,p in mixer.named_parameters()}
        def capture(name,module,args,output):
            values=output if isinstance(output,tuple) else (output,)
            for j,value in enumerate(values):
                if not isinstance(value,torch.Tensor) or value.ndim not in (2,3) or self.span is None:
                    continue
                # Only [tokens,width] or [batch,sequence,width], never [batch,width,sequence].
                row_count=value.numel()//value.shape[-1]
                if row_count < self.span[1]: continue
                key=f'L{layer:02d}_detail_{name.replace(".","-")}_{j}'
                self.keep(key,f'{base}.{layer}.mixer.{name}',value)
                self.detail_observability['observed'].append(name)
        for name,module in mixer.named_modules():
            if name and name.split('.')[-1] in wanted:
                # Convolution layout differs; retain its source/dtype in runtime configuration instead.
                if isinstance(module,nn.Conv1d):
                    self.detail_observability['convolution']='functional/channel-major convolution requires kernel-level replay'
                    continue
                self.detail_handles.append(module.register_forward_hook(partial(capture,name)))
        router=getattr(getattr(mixer,'experts',None),'router',None)
        select=getattr(router,'select_experts',None)
        if callable(select):
            def wrapped(*args,**kwargs):
                result=select(*args,**kwargs)
                capture('actual_router_selection',router,args,result)
                self.detail_observability['routing_ids']='captured select_experts return; tensor dtypes identify IDs and weights'
                return result
            router.select_experts=wrapped
            self.restore_methods.append((router,'select_experts',select))

    def post(self, key, path, module, args, output):
        self.keep(key,path,tensor_output(output))

    def reset(self):
        self.values.clear()
        self.span = None

    def close(self):
        for owner,name,method in self.restore_methods:
            setattr(owner,name,method)
        for handle in self.detail_handles:
            handle.remove()
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.reset()


class DiagnosticWorker:
    """Worker installed only by the diagnostic script, discoverable in the wheel."""
    def diag_install(self, reference):
        self.diag_trace = TraceHooks(self.model_runner.model, worker=True)
        self.diag_case = None
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
            from .hooks import representative_sites
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
            rid = internal
            if rid not in case['requests']:
                external,sep,suffix = rid.rpartition('-')
                if not sep or len(suffix)!=8 or external not in case['requests']:
                    raise RuntimeError('Unknown request ID '+rid)
                rid=external
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
        partial_path=path.with_suffix('.safetensors.partial')
        save_file(tensors,str(partial_path));partial_path.replace(path)
        for key,value in load_file(str(path)).items():
            if not torch.equal(value,tensors[key]):
                raise RuntimeError('Trace persistence mismatch '+key)
        report=dict(name=case['name'],scheduler=case['scheduler'],same_forward=case['equality'],
            topology=self.diag_trace.topology,path=str(path),trace_persistence_exact=True,
            operator_observability=self.diag_trace.detail_observability)
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
        if getattr(self,'diag_sibling',None) is not None and self.diag_case is not None:
            self.diag_sibling.sae_remove_hooks()
        for handle in getattr(self,'diag_package_handles',[]):
            handle.remove()
        for handle in self.diag_handles:
            handle.remove()
        self.diag_trace.close()
        self.diag_case=None


class IntegrityWorker(ActivationHookExtension):
    """Verify production asynchronous copies against same-forward GPU snapshots."""
    def nh_install(self,sites,sink_dir):
        result=super().nh_install(sites,sink_dir)
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
        wave['diagnostic_occupied_slots']=2-len(self.nh_pool.free)

    @staticmethod
    def _nh_publish(wave):
        # Only diagnostic workers delay publication to exercise the second slot.
        time.sleep(.5)
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
