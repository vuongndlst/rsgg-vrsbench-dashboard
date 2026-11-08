"""One finite TRAIN-only GPU benchmark, run exclusively inside the own 8 GiB cap.

Never changes locked v2 sources, checkpoints, caches, criterion, or environment.
Microbatch changes dropout RNG trajectories, so adoption requires a new matched
experiment, even if eval-mode output/loss/gradient/update parity passes.
"""
from pathlib import Path
from datetime import datetime, timezone
import gc, hashlib, json, os, resource, shutil, signal, subprocess, sys, time
os.environ.update(OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='4')
root=Path.home()/'rsg-former'
folder=root/'artifacts/caption_readiness/vg_speedup_benchmark_v1'
assert not folder.exists(), 'preserve_finite_benchmark_no_hidden_retry'
assert shutil.disk_usage(root).free > 16*1024**3 + 256*1024**2
lock=json.loads((root/'reports/VG_V2_PREREGISTRATION.json').read_text())
sources=dict(lock['source_hashes'])
sources.update({'vendor/salience_sgg/models/salience_sgg/deformable_detr.py':'b79111a2ce07827915b37b6991e2904e2445ff701db30f7cac4be01d4783d67a',
                'vendor/salience_sgg/models/salience_sgg/deformable_transformer.py':'16ab624681c3a43686346cb07be8b0d178c7535ac78ff182c1ff3abe9ef27558'})
for name in ['src/rsgformer/vg_speedup.py','scripts/remote_vg_speedup_benchmark.py']:
    sources[name]=hashlib.sha256((root/name).read_bytes()).hexdigest()
for name,h in sources.items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==h
pid=int((root/'artifacts/vg_method_v2/pipeline.pid').read_text())
cmdline=Path('/proc')/str(pid)/'cmdline'
assert not cmdline.exists() or not cmdline.read_bytes(), 'wait_for_exact_v2_process_exit'
manifest_path=root/'artifacts/vg_method_v2/features/manifest.json'
manifest_raw=manifest_path.read_bytes(); manifest=json.loads(manifest_raw)
assert manifest['status']=='complete' and manifest['cache_dtype']=='float32'
assert manifest['preregistration_source_hashes']==lock['source_hashes']
entry=manifest['splits']['train'][0]
protocol={'utc':datetime.now(timezone.utc).isoformat(), 'source_hashes':sources,
          'manifest_sha256':hashlib.sha256(manifest_raw).hexdigest(),
          'selection':'first 32 images of first locked TRAIN shard; no metric selection',
          'shard':entry,'seed':1729,'effective_batch':16,'warmup_images':16,'timed_images':16,
          'variants':['native','context','oneway','feedback'],
          'cases':['original_fp32_m1','packed_fp32_m1','packed_fp32_m2','packed_fp32_m4'],
          'parity':{'mode':'eval with gradients, first four train images',
                    'output_rtol':1e-5,'output_atol':1e-5,
                    'loss_rtol':1e-5,'loss_atol':1e-5,
                    'gradient_rtol':1e-4,'gradient_atol':1e-5,
                    'update_rtol':1e-4,'update_atol':1e-6},
          'selection_rule':'all four arms pass parity/resource; minimum arm speedup >=1.10; choose highest minimum speedup; otherwise keep original',
          'training_adoption':'new experiment for microbatch>1; all four arms restart from identical original initialization; no reuse of old native as matched new control',
          'resource':{'cpu_threads':4,'own_gpu_nvml_bytes_cap':8*1024**3,
                      'cuda_allocator_gib':6,'peak_rss_gib':4,'timeout_seconds':240,
                      'minimum_free_disk_gib':16,'minimum_gpu_free_gib':9},
          'benchmark_test_accessed':False,'validation_scored':False,'live_sources_modified':False}
folder.mkdir(); raw=json.dumps(protocol,indent=2).encode()
(folder/'preregistration.json').write_bytes(raw)
report={'status':'FAILED','protocol':protocol,'preregistration_sha256':hashlib.sha256(raw).hexdigest(),
        'benchmark_test_accessed':False,'validation_scored':False,'training_adopted':False}
try:
    signal.alarm(240)
    cpus=sorted(os.sched_getaffinity(0)); os.sched_setaffinity(0,set(cpus[:4]))
    import torch, numpy as np
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('highest')
    torch.cuda.set_per_process_memory_fraction(6*1024**3/torch.cuda.get_device_properties(0).total_memory)
    free,_=torch.cuda.mem_get_info();assert free>=9*1024**3,'shared_gpu_free_admission'
    from rsgformer.vg_v2_runtime import native_model,datasets,Adaptation
    from rsgformer.vg_speedup import pack,target,per_image_losses,finite_loss
    torch.manual_seed(1729)
    model,criterion,_=native_model(root);model.cuda().eval()
    ds=datasets(root,'train');assert ds.split_hash()==lock['split_hashes']['train']
    path=manifest_path.parent/entry['path']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256']
    chunk=torch.load(path,map_location='cpu',weights_only=True)
    assert [x['image_id'] for x in chunk]==entry['image_ids']
    items=chunk[:32];del chunk
    assert len(items)==32 and all(ds[x['index']].image_id==x['image_id'] for x in items)
    initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items() if 'add_add_add_relation' in k}
    def guard():
        assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=4*1024**3,'cpu_rss_cap'
        rows=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5)
        assert rows.returncode==0,'nvml_admission_unavailable'
        own=[int(row.split(',')[1].strip())*1024**2 for row in rows.stdout.splitlines() if int(row.split(',')[0])==os.getpid()]
        assert len(own)==1 and own[0]<=8*1024**3,'own_process_nvml_cap'
        return own[0]
    def init(variant):
        model.load_state_dict(initial,strict=False);torch.manual_seed(1729)
        a=Adaptation(model,variant).cuda()
        return a, [p for p in a.parameters() if p.requires_grad]
    def original_pack(item):
        return {k:item[k][None].float().cuda() for k in ('nodes','boxes','logits','visual')}
    def compute(a, selected, micro, original=False, output_store=None):
        losses=[]
        for start in range(0,len(selected),micro):
            group=selected[start:start+micro]
            b=original_pack(group[0]) if original else pack(group,pinned=True)
            pred,_=a(b)
            if output_store is not None:
                from rsgformer.vg_speedup import singleton
                output_store.extend([{k:v.detach().cpu().clone() for k,v in singleton(pred,i).items()} for i in range(len(group))])
            values=per_image_losses(criterion,pred,[target(ds[x['index']]) for x in group])
            finite_loss(values)
            (values.sum()/16).backward()
            losses.extend(values.detach().cpu().tolist())
            del b,pred,values
        return losses
    cases={};parities={}
    for variant in protocol['variants']:
        reference=None;parities[variant]={};cases[variant]={}
        for micro in [1,2,4]:
            a,params=init(variant);a.eval();opt=torch.optim.AdamW(params,lr=1e-4,weight_decay=1e-4)
            try:
                outputs=[]
                values=compute(a,items[:4],micro,original=micro==1,output_store=outputs)
                grads={n:p.grad.detach().cpu().clone() for n,p in a.named_parameters() if p.requires_grad and p.grad is not None}
                assert grads and all(bool(torch.isfinite(g).all()) for g in grads.values()),'non_finite_gradient'
                torch.nn.utils.clip_grad_norm_(params,.1);opt.step()
                updated={n:p.detach().cpu().clone() for n,p in a.named_parameters() if p.requires_grad}
                result={'status':'PASS','gradient_tensors':len(grads),'max_output_error':0.,'max_gradient_error':0.,'max_update_error':0.,'losses':values}
                if reference is None: reference=(values,grads,updated,outputs)
                else:
                    for actual,expected in zip(outputs,reference[3]):
                        assert actual.keys()==expected.keys()
                        for k,v in actual.items():
                            torch.testing.assert_close(v,expected[k],rtol=1e-5,atol=1e-5,msg=k)
                            result['max_output_error']=max(result['max_output_error'],float((v-expected[k]).abs().max()))
                    torch.testing.assert_close(torch.tensor(values),torch.tensor(reference[0]),rtol=1e-5,atol=1e-5)
                    assert grads.keys()==reference[1].keys()
                    for n,g in grads.items():
                        torch.testing.assert_close(g,reference[1][n],rtol=1e-4,atol=1e-5,msg=n)
                        result['max_gradient_error']=max(result['max_gradient_error'],float((g-reference[1][n]).abs().max()))
                    for n,v in updated.items():
                        torch.testing.assert_close(v,reference[2][n],rtol=1e-4,atol=1e-6,msg=n)
                        result['max_update_error']=max(result['max_update_error'],float((v-reference[2][n]).abs().max()))
                result['observed_nvml_bytes']=guard()
            except (AssertionError,torch.cuda.OutOfMemoryError) as e:
                result={'status':'REJECTED','error_type':type(e).__name__,'reason':str(e)[:600]}
            parities[variant][str(micro)]=result
            a.zero_grad(set_to_none=True);model.zero_grad(set_to_none=True)
            del a,params,opt
            if 'grads' in locals():del grads
            if 'updated' in locals():del updated
            if 'outputs' in locals():del outputs
            gc.collect();torch.cuda.empty_cache()
        del reference
        for case,micro,original in [('original_fp32_m1',1,True),('packed_fp32_m1',1,False),('packed_fp32_m2',2,False),('packed_fp32_m4',4,False)]:
            if parities[variant][str(micro)]['status']!='PASS':
                cases[variant][case]={'status':'NOT_TIMED_PARITY_REJECTED'};continue
            a,params=init(variant);a.train();opt=torch.optim.AdamW(params,lr=1e-4,weight_decay=1e-4)
            try:
                rounds=[]
                for repeat in range(2):
                    a.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
                    started=time.monotonic()
                    values=compute(a,items[16*repeat:16*(repeat+1)],micro,original=original)
                    torch.nn.utils.clip_grad_norm_(params,.1);opt.step();torch.cuda.synchronize()
                    rounds.append({'seconds':time.monotonic()-started,'images':16,'loss_mean':sum(values)/16,
                                   'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'observed_nvml_bytes':guard()})
                cases[variant][case]={'status':'PASS','warmup':rounds[0],'timed':rounds[1],'seconds_per_image':rounds[1]['seconds']/16}
            except (AssertionError,torch.cuda.OutOfMemoryError) as e:
                cases[variant][case]={'status':'REJECTED','error_type':type(e).__name__,'reason':str(e)[:600]}
            a.zero_grad(set_to_none=True);model.zero_grad(set_to_none=True)
            del a,params,opt;gc.collect();torch.cuda.empty_cache()
    accepted={}
    for case in protocol['cases'][1:]:
        ratios=[]
        for variant in protocol['variants']:
            base=cases[variant]['original_fp32_m1'];candidate=cases[variant][case]
            if base['status']!='PASS' or candidate['status']!='PASS':break
            ratios.append(base['seconds_per_image']/candidate['seconds_per_image'])
        if len(ratios)==4 and min(ratios)>=1.10:accepted[case]={'minimum_arm_speedup':min(ratios),'speedup_by_arm':dict(zip(protocol['variants'],ratios))}
    selected=max(accepted,key=lambda k:accepted[k]['minimum_arm_speedup']) if accepted else None
    report.update(status='BENCHMARK_COMPLETE',cases=cases,parity=parities,accepted=accepted,selected_case=selected,
                  adoption_ready=selected is not None,training_adopted=False,
                  limits='One timed optimizer update per arm/case; GPU shared; no full-epoch or validation accuracy guarantee; train dropout RNG trajectory differs across microbatches.')
except Exception as e:
    import traceback
    report.update(status='BENCHMARK_FAILED',error_type=type(e).__name__,reason=str(e)[:1000],traceback=traceback.format_exc()[-3000:])
finally:
    signal.alarm(0)
    report['utc']=datetime.now(timezone.utc).isoformat()
    report['source_lock_still_verified']=all(hashlib.sha256((root/p).read_bytes()).hexdigest()==h for p,h in sources.items())
    text=json.dumps(report,indent=2).replace(str(Path.home()),'${REMOTE_HOME}')
    (folder/'outcome.json').write_text(text)
    (root/'reports/VG_SPEEDUP_BENCHMARK.json').write_text(text)
    print(text)
