# /// script
# requires-python = ">=3.10"
# dependencies = ["nemotron-extraction[hf,vllm,benchmark]==1.1.0"]
# ///
"""Focused extraction benchmark. Install the selected frozen wheel with uv first.

Run each method/batch in its own process, sequentially. Comparison is a separate
CPU stage; detailed profiling is a separate run. No training or cluster settings.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import tempfile
import time
import traceback

import torch
from safetensors import safe_open
from safetensors.torch import save_file
import nemotron_extraction
from nemotron_extraction import HFExtractor, VLLMExtractor, HookSite
from nemotron_extraction.hooks import representative_sites
from nemotron_extraction.worker import ActivationHookExtension
from compare_repositories import (SiblingExtractor, configured_kv_cache_bytes, phase,
    persist, sibling_namespace)
from diagnose_divergence import inputs, REVISION
from targeted_validation import Utilisation
from validate import Agreement, sha256, versions, write_json

METHODS = ('transformers', 'package_baseline', 'package_optimized', 'sibling_files', 'sibling_fills')


class BaselineWorker(ActivationHookExtension):
    """Add scheduler receipts to the frozen baseline without changing captures."""
    def _nh_inputs(self,module,args,kwargs):
        super()._nh_inputs(module,args,kwargs)
        wave=self.nh_active
        reverse={v:k for k,v in wave['requests'].items()}
        wave.setdefault('scheduler',[]).extend(dict(request_id=reverse[i],sequence_index=i,
            start=start,stop=stop,computed_tokens=0,prompt_length=wave['tokens'].shape[1])
            for i,start,stop in self.nh_spans)

    @staticmethod
    def _nh_publish(wave):
        result=ActivationHookExtension._nh_publish(wave)
        result['scheduler']=wave['scheduler']
        return result


class BaselineExtractor(VLLMExtractor):
    _worker_extension_cls='extraction_speed.BaselineWorker'


def prune_receipts(measurement,root):
    """Remove only capture files explicitly named by a verified measurement."""
    root=Path(root).resolve()
    for receipt in measurement['receipts']:
        paths=receipt.get('paths')
        if paths is None:
            path=Path(receipt['path'])
            paths=list(path.glob('*.safetensors')) if path.is_dir() else [path]
        for path in paths:
            path=Path(path).resolve()
            if root not in path.parents: raise ValueError('Capture path is outside configuration directory')
            path.unlink()
    measurement['captures_pruned']=True


def workload(rows, batch_size):
    if rows.shape != (64, 2048) or batch_size not in (8, 64):
        raise ValueError('Requires 64 frozen 2048-token sequences and batch 8 or 64')
    return list(rows.split(batch_size))


def destination(args):
    suffix = f'{args.method}-t{args.publication_threads}'
    if args.profile: suffix += '-profile'
    return args.output / f'b{args.batch_size}' / suffix


def configuration(args):
    return dict(enforce_eager=True, tensor_parallel_size=1, enable_prefix_caching=False,
        enable_chunked_prefill=True, max_model_len=2049, max_num_seqs=args.batch_size,
        max_num_batched_tokens=args.batch_size*2049, gpu_memory_utilization=.96,
        kv_cache_memory_bytes=configured_kv_cache_bytes())


def check_write_budget(root,additional_bytes):
    limit=os.environ.get('NH_RUN_STORAGE_LIMIT_BYTES')
    if limit:
        used=sum(p.stat().st_size for p in root.rglob('*') if p.is_file() and not p.is_symlink())
        if used+additional_bytes>int(limit):
            raise RuntimeError(f'Run storage budget exceeded: {used} + {additional_bytes} > {limit}')


def run(args):
    torch.set_num_threads(4)
    torch.manual_seed(0)
    manifest, rows = inputs(args.work)
    batches = workload(rows, args.batch_size)
    target = destination(args)
    target.mkdir(parents=True, exist_ok=True)
    receipt_path = target/'result.json'
    if receipt_path.exists():
        old = json.loads(receipt_path.read_text())
        if old['status'] == 'complete':
            print('Preserving complete configuration', target, flush=True)
            return 0
        raise FileExistsError('Preserve failed configuration; use a new output directory')
    record = dict(status='running', method=args.method, batch_size=args.batch_size,
        publication_threads=args.publication_threads, profile=args.profile, command=sys.argv,
        model_revision=REVISION, token_sha256=manifest['languages']['en']['token_sha256'],
        imported_package=str(nemotron_extraction.__file__), software=versions(),
        source_hashes={p.name:sha256(p) for p in sorted(Path(nemotron_extraction.__file__).parent.glob('*.py'))},
        sibling_hashes={p.name:sha256(p) for p in sorted(args.reference.glob('*.py'))},
        script_sha256=sha256(__file__), engine_configuration=configuration(args),
        gpu=torch.cuda.get_device_name(), repetitions=[],
        timing_semantics='GPU forward includes capture hooks; overlapping stages must not be added. '
                         'Sibling files include original driver readback and assembly.')
    extractor = None
    monitor = Utilisation(target/'utilisation.jsonl')
    started = time.perf_counter()
    current_phase = 'startup'
    try:
        with monitor:
            sites_path = args.output/'sites.json'
            if args.method == 'transformers':
                from transformers import AutoModelForCausalLM
                model = AutoModelForCausalLM.from_pretrained(manifest['model_path'],
                    torch_dtype=torch.bfloat16, device_map='cuda', trust_remote_code=True, local_files_only=True)
                sites = representative_sites(model)
                if len(sites) != 18: raise ValueError('Expected all 18 representative sites')
                resolved = [dict(layer=s.layer,kind=s.kind) for s in sites]
                if sites_path.exists() and json.loads(sites_path.read_text()) != resolved:
                    raise ValueError('Resolved site set changed')
                write_json(sites_path, resolved)
                extractor = HFExtractor(model, sites, diagnostics=args.profile,
                                        publication_threads=args.publication_threads)
            else:
                sites = [HookSite(**s) for s in json.loads(sites_path.read_text())]
                if args.method.startswith('package_'):
                    options = dict(sequences_per_call=args.batch_size, max_model_len=2049,
                        enable_chunked_prefill=True, gpu_memory_utilization=.96,
                        kv_cache_memory_bytes=configured_kv_cache_bytes(), sink_dir=str(target))
                    if args.method == 'package_optimized':
                        options.update(diagnostics=args.profile,publication_threads=args.publication_threads)
                    elif args.publication_threads != 1 or args.profile:
                        raise ValueError('Baseline wheel is measured unchanged with one writer')
                    factory=BaselineExtractor if args.method=='package_baseline' else VLLMExtractor
                    extractor = factory(manifest['model_path'], sites, **options)
                else:
                    extractor = SiblingExtractor(manifest['model_path'], sites, args.reference,
                        args.method, args.batch_size, 2048, target/'raw')
            record.update(startup_seconds=time.perf_counter()-started,
                hook_install_seconds=extractor.hook_install_seconds)
            current_phase = monitor.phase = 'warmup'
            widths_path=args.output.parent/'widths.json'
            if os.environ.get('NH_RUN_STORAGE_LIMIT_BYTES'):
                widths=json.loads(widths_path.read_text())
                warmup_payload=2*batches[0].numel()*(sum(widths[s.key] for s in sites)*2+8)
                check_write_budget(args.output.parent,warmup_payload+10_000_000)
            with tempfile.TemporaryDirectory(dir=target) as temp:
                warm_started=time.perf_counter()
                asyncio.run(persist(extractor,[batches[0],batches[0]],Path(temp)))
                record['warmup_seconds']=time.perf_counter()-warm_started
            # Original file sink puts its captures under its own sink directory.
            if args.method == 'sibling_files':
                for p in extractor.sink.glob('step_*.safetensors'): p.unlink()
            for repeat in range(1 if args.profile else args.repetitions):
                current_phase = monitor.phase = f'repetition-{repeat}'
                stage=target/f'rep-{repeat}'
                payload=rows.numel()*(sum(s['width'] for s in extractor.topology.values())*2+8)
                check_write_budget(args.output.parent,payload+10_000_000)
                measurement=phase(extractor,batches if not args.profile else [batches[0],batches[0]],stage)
                measurement.update(repetition=repeat,
                    validation_seconds=sum(r.get('validation_seconds',0.) for r in measurement['receipts']),
                    serialization_seconds=sum(r.get('serialization_seconds',0.) for r in measurement['receipts']),
                    capture_seconds=(sum(r['capture_seconds'] for r in measurement['receipts'])
                        if args.profile and all(r.get('capture_seconds') is not None for r in measurement['receipts']) else None))
                scheduler=measurement.get('scheduler',[])+[s for r in measurement['receipts'] for s in r.get('scheduler',[])]
                if args.method != 'transformers':
                    if len(scheduler) != (2*args.batch_size if args.profile else 64):
                        raise ValueError('Missing whole-prefill scheduler receipts')
                    if any(s['computed_tokens'] != 0 or s['stop']-s['start'] != 2048 for s in scheduler):
                        raise ValueError('Partial prefill in timing workload')
                measurement['scheduler']=scheduler
                measurement['memory_peaks']={name:getattr(extractor,name,None) for name in
                    ('peak_gpu_memory_bytes','peak_gpu_reserved_bytes','peak_device_memory_bytes',
                     'total_device_memory_bytes','peak_device_memory_fraction')}
                record['repetitions'].append(measurement)
                write_json(stage/'receipt.json',measurement)
                write_json(receipt_path,record)
                if not args.profile:
                    current_phase=monitor.phase=f'comparison-{repeat}'
                    compare_started=time.perf_counter()
                    is_reference=args.method=='transformers' and args.batch_size==8
                    reference_record=(record if is_reference else json.loads(
                        (args.output/'b8/transformers-t1/result.json').read_text()))
                    measurement['agreement']=compare_measurements(reference_record['repetitions'][0],
                        measurement,rows,sites,args.reference,stage/'agreement')
                    if repeat:
                        measurement['repeat_agreement']=compare_measurements(record['repetitions'][0],
                            measurement,rows,sites,args.reference,stage/'repeat-agreement')
                    measurement['comparison_seconds']=time.perf_counter()-compare_started
                    write_json(stage/'agreement.json',dict(reference=measurement['agreement'],
                        repetition=measurement.get('repeat_agreement')))
                    # Keep repetition zero until repeatability checks finish.
                    if repeat: prune_receipts(measurement,target)
                    write_json(receipt_path,record)
            record['topology']=extractor.topology
            if not args.profile and not (args.method=='transformers' and args.batch_size==8):
                prune_receipts(record['repetitions'][0],target)
            record['status']='complete'
    except Exception as error:
        message=str(error)
        record.update(status='memory_failure' if isinstance(error,torch.OutOfMemoryError) or
            any(s in message.lower() for s in ('out of memory','no available memory','kv cache blocks')) else 'error',
            error=message,traceback=traceback.format_exc(),failed_phase=current_phase)
    finally:
        if extractor is not None:
            try: extractor.close()
            except Exception as error:
                record['cleanup_error']=str(error);record['status']='error'
    record.update(process_wall_seconds=time.perf_counter()-started,utilisation=monitor.summary())
    write_json(receipt_path,record)
    print(json.dumps({k:record.get(k) for k in ('status','method','batch_size','error','process_wall_seconds')}),flush=True)
    return 0 if record['status']=='complete' else 1


def read_site(receipt,tokens,key,snapshot):
    if receipt.get('mode') == 'sibling_files':
        steps=[]
        for path in receipt['paths']:
            with safe_open(path,framework='pt',device='cpu') as f:
                steps.append({k:f.get_tensor(k) for k in ('input_ids','positions',key)})
        return sibling_namespace(snapshot)['assemble_rows'](steps,tokens,[key])[key]
    if receipt.get('mode') == 'sibling_fills':
        root=Path(receipt['path'])
        with safe_open(root/'token_ids.safetensors',framework='pt',device='cpu') as f:
            if not torch.equal(f.get_tensor('token_ids'),tokens): raise ValueError('Token attribution differs')
        with safe_open(root/f'{key}.safetensors',framework='pt',device='cpu') as f:
            return f.get_tensor('activations')
    with safe_open(receipt['path'],framework='pt',device='cpu') as f:
        if not torch.equal(f.get_tensor('token_ids'),tokens): raise ValueError('Token attribution differs')
        return f.get_tensor(key)


def compare_measurements(left,right,rows,sites,snapshot,output):
    """Align different shard boundaries; hold at most one site from each shard."""
    output.mkdir(parents=True,exist_ok=True)
    results={}
    for site in sites:
        metric=Agreement(); exact=True; start=0
        left_hash,right_hash=hashlib.sha256(),hashlib.sha256()
        candidates=iter(right['receipts']); candidate=None; offset=0
        candidate_start=0
        for receipt in left['receipts']:
            count=receipt['tokens'];tokens=rows.flatten()[start:start+count].reshape(-1,2048)
            reference=read_site(receipt,tokens,site.key,snapshot);pos=0
            if len(reference)!=count: raise ValueError('Reference row count differs')
            while pos<count:
                if candidate is None or offset==len(candidate):
                    r=next(candidates);n=r['tokens']
                    t=rows.flatten()[candidate_start:candidate_start+n].reshape(-1,2048)
                    candidate=read_site(r,t,site.key,snapshot);offset=0;candidate_start+=n
                    if len(candidate)!=n: raise ValueError('Candidate row count differs')
                take=min(count-pos,len(candidate)-offset)
                a,b=reference[pos:pos+take],candidate[offset:offset+take]
                left_hash.update(a.contiguous().view(torch.uint8).numpy().tobytes())
                right_hash.update(b.contiguous().view(torch.uint8).numpy().tobytes())
                metric.add(a,b);exact=exact and torch.equal(a,b)
                if start==0 and pos==0:
                    save_file({'reference':a[:8].clone(),'candidate':b[:8].clone()},str(output/(site.key+'.safetensors')))
                pos+=take;offset+=take
            start+=count
        if start!=rows.numel() or candidate_start!=start or offset!=len(candidate) or next(candidates,None) is not None:
            raise ValueError('Comparison token counts differ')
        r=metric.result();r.update(exact=exact,cosine_std=float(torch.cat(metric.cosines).std(correction=0)),
            reference_sha256=left_hash.hexdigest(),candidate_sha256=right_hash.hexdigest())
        results[site.key]=r
    return results


def compare(args):
    _,rows=inputs(args.work)
    sites=[HookSite(**s) for s in json.loads((args.output/'sites.json').read_text())]
    records=[(p,json.loads(p.read_text())) for p in sorted(args.output.glob('b*/*/result.json')) if '-profile' not in str(p.parent)]
    reference=next((r for _,r in records if r['method']=='transformers' and r['batch_size']==8 and r['status']=='complete'),None)
    if reference is None: raise ValueError('Completed batch-8 Transformers reference required')
    started=time.perf_counter();results=[]
    for path,r in records:
        if r['status']!='complete':
            results.append(dict(configuration=str(path.parent.relative_to(args.output)),status=r['status'],error=r.get('error')));continue
        for key in ('model_revision','token_sha256'):
            if r[key]!=reference[key]: raise ValueError('Comparison provenance differs: '+key)
        # Compare every repetition; exact comparisons against repetition zero
        # additionally expose run-to-run variation without assuming determinism.
        agreements=[m['agreement'] for m in r['repetitions']]
        repeat_agreement=[m['repeat_agreement'] for m in r['repetitions'][1:]]
        write_json(path.parent/'agreement.json',dict(reference=agreements,repetitions=repeat_agreement))
        speeds=[m['completed_to_disk_tokens_per_second'] for m in r['repetitions']]
        result=dict(configuration=str(path.parent.relative_to(args.output)),status='complete',
            completed_to_disk_tokens_per_second_median=statistics.median(speeds),
            completed_to_disk_tokens_per_second_min=min(speeds),completed_to_disk_tokens_per_second_max=max(speeds),
            passed_sites=[sum(s['passed'] for s in a.values()) for a in agreements],agreement=agreements,
            startup_seconds=r['startup_seconds'],warmup_seconds=r['warmup_seconds'])
        results.append(result)
    for result in results:
        if 'package_optimized' not in result['configuration'] or result['status']!='complete': continue
        batch=result['configuration'].split('/')[0]
        baseline=next((r for r in results if r['configuration']==batch+'/package_baseline-t1' and r['status']=='complete'),None)
        if baseline:
            result['exact_vs_baseline']=[all(a[key]['candidate_sha256']==b[key]['candidate_sha256']
                for key in a) for a,b in zip(result['agreement'],baseline['agreement'])]
    write_json(args.output/'comparison.json',dict(results=results,comparison_seconds=time.perf_counter()-started,
        numerical_gates=dict(mean_cosine=.9999,minimum_cosine=.9990,relative_l2=.02,normalized_mse=.0004)))
    lines=['# Extraction comparison','','Throughput counts prompt activations saved to disk after warmup. Numerical agreement is assessed separately against Transformers.','',
        '| Configuration | Median tokens/s | Range | Passing sites/repetition |','|---|---:|---:|---|']
    for r in results:
        if r['status']=='complete': lines.append(f"| {r['configuration']} | {r['completed_to_disk_tokens_per_second_median']:.0f} | {r['completed_to_disk_tokens_per_second_min']:.0f}–{r['completed_to_disk_tokens_per_second_max']:.0f} | {r['passed_sites']} |")
        else: lines.append(f"| {r['configuration']} | {r['status']} | | |")
    (args.output/'comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('run','compare'))
    parser.add_argument('--work',type=Path,required=True)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--method',choices=METHODS)
    parser.add_argument('--batch-size',type=int,choices=(8,64),default=8)
    parser.add_argument('--publication-threads',type=int,choices=(1,2),default=1)
    parser.add_argument('--repetitions',type=int,choices=(1,3),default=3)
    parser.add_argument('--profile',action='store_true')
    args=parser.parse_args()
    if args.stage=='run' and args.method is None: parser.error('--method required for run')
    torch.set_num_threads(4)
    return run(args) if args.stage=='run' else compare(args)


if __name__=='__main__':
    raise SystemExit(main())
