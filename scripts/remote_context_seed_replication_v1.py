"""Human24Oct: fixed Context seeds2718/3141; train/validation-only replication."""
from pathlib import Path
import os,sys,json,time,datetime,hashlib,subprocess,traceback,shutil,random,io,resource
ROOT=Path.home()/'rsg-former';ART=ROOT/'artifacts/vg_context_seed_replication_v1'
CFG=json.loads((ROOT/'configs/vg_context_seed_replication_v1.json').read_text());START=time.time()
PREREG=json.loads((ART/'preregistration.json').read_text())
ORIGINAL_START=1792510321.2414355-240*3600

def atomic(path,value):
 t=path.with_suffix(path.suffix+'.tmp');t.write_text(json.dumps(value,indent=2));t.replace(path)
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for part in iter(lambda:f.read(8*1024*1024),b''):h.update(part)
 return h.hexdigest()
def verify():
 assert all(sha(ROOT/p)==h for p,h in PREREG['source_hashes'].items()),'immutable_source_changed'
def event(stage,**kw):
 o={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'experiment_id':CFG['experiment_id'],
    'stage':stage,'wall_seconds':time.time()-ORIGINAL_START,'version_wall_seconds':time.time()-START,**kw}
 atomic(ROOT/'reports/VG_CONTEXT_SEED_REPLICATION_V1_PROGRESS.json',o);print(json.dumps(o),flush=True)
def guard():
 if (ART/'STOP').exists():raise RuntimeError('user_stop_flag')
 if time.time()>=CFG['authorized_wall_deadline_unix']:raise RuntimeError('authorized_extended_wall_budget_reached')
 if shutil.disk_usage(ROOT).free<16*1024**3:raise RuntimeError('disk_reserve_reached')
def resources():
 guard()
 assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=4*1024**3,'cpu_rss_cap'
 q=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5)
 assert q.returncode==0,'nvml_unavailable'
 own=[int(row.split(',')[1])*1024**2 for row in q.stdout.splitlines() if int(row.split(',')[0])==os.getpid()]
 assert len(own)==1 and own[0]<=8*1024**3,'own_gpu_process_cap'
 return {'nvml_bytes':own[0],'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
def artifact_guard():
 total=sum(p.stat().st_size for base in [ROOT/'artifacts/vg_method_v2',ROOT/'artifacts/vg_method_v2_speed_m2_v1',ROOT/'artifacts/vg_component_control_test_v1',ART] for p in base.rglob('*') if p.is_file())
 assert total<=40*1024**3,'aggregate_artifact_budget_reached'
 own=sum(p.stat().st_size for p in ART.rglob('*') if p.is_file())
 assert own<=2*1024**3,'new_phase_artifact_cap'

def main():
 verify();guard()
 for name,h in PREREG['inherited_screening_file_hashes'].items():assert sha(ROOT/name)==h,'inherited_screening_evidence_changed'
 event('resource_admission',execution_revision=CFG['execution_revision'])
 cpus=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,set(cpus[:4]))
 import torch,numpy as np
 from rsgformer.vg_v2_runtime import native_model,datasets,capture,Adaptation,SGDetEvaluator,predicted_graph
 from rsgformer.vg_speedup import pack,target,per_image_losses,finite_loss
 from rsgformer.vg_speedup_schedule import grouped,step_boundary
 torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.set_float32_matmul_precision('highest')
 torch.cuda.set_per_process_memory_fraction(6*1024**3/torch.cuda.get_device_properties(0).total_memory)
 assert torch.cuda.mem_get_info()[0]>=9*1024**3,'shared_gpu_free_admission'
 torch.manual_seed(1729)
 model,criterion,_=native_model(ROOT);model.cuda().eval();resources()
 train=datasets(ROOT,'train');val=datasets(ROOT,'val')
 assert train.split_hash()==PREREG['split_hashes']['train'] and val.split_hash()==PREREG['split_hashes']['val']
 prereg=PREREG
 cache=ROOT/'artifacts/vg_method_v2/features';cache_manifest=cache/'manifest.json'
 assert sha(cache_manifest)==PREREG['cache_manifest_sha256'],'immutable_cache_manifest_changed'
 manifest=json.loads(cache_manifest.read_text())
 assert manifest['status']=='complete' and manifest['cache_dtype']=='float32'
 def items(split,rng=None):
  assert split in ('train','val'),'test_access_forbidden'
  entries=list(manifest['splits'][split])
  if rng is not None:rng.shuffle(entries)
  ds=train if split=='train' else val
  for e in entries:
   guard();p=cache/e['path'];raw=p.read_bytes()
   assert hashlib.sha256(raw).hexdigest()==e['sha256'],'cache_shard_hash_changed'
   chunk=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True);del raw
   assert [x['image_id'] for x in chunk]==e['image_ids'],'cache_shard_membership_changed'
   if rng is not None:rng.shuffle(chunk)
   for item in chunk:
    assert ds[item['index']].image_id==item['image_id'],'cache_annotation_index_changed'
    assert all(item[k].dtype==torch.float32 for k in ['nodes','boxes','logits','visual'])
    yield item
 def batch(item):return {k:item[k][None].float().cuda() for k in ['nodes','boxes','logits','visual']}
 def evaluate(adapt,split,ds,records_path=None):
  adapt.eval();model.eval();ev=SGDetEvaluator();start=time.time()
  with torch.no_grad():
   for j,item in enumerate(items(split)):
    guard();sample=ds[item['index']];pred,_=adapt(batch(item));ev.update(sample,*predicted_graph(pred,sample))
    if (j+1)%128==0:event('validation',split=split,images=j+1,seconds_per_image=(time.time()-start)/(j+1),**resources())
  if records_path:atomic(records_path,ev.records)
  return ev.result()
 # Reuse only the immutable frozen validation baseline, never an adapted head.
 for name,h in PREREG['frozen_baseline_source_hashes'].items():assert sha(ROOT/name)==h
 frozen=json.loads((ROOT/'artifacts/vg_method_v2/frozen_native_validation.json').read_text())
 # Read existing frozen baseline; preserve original completed evidence files.
 initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items() if 'add_add_add_relation' in k}
 initial_digest=hashlib.sha256()
 for k,v in sorted(initial.items()):
  initial_digest.update(k.encode());initial_digest.update(str(v.dtype).encode());initial_digest.update(str(tuple(v.shape)).encode());initial_digest.update(v.numpy().tobytes())
 initial_sha=initial_digest.hexdigest()
 assert initial_sha=='bc2ceb642f57f26729b104da880f9e5057c3b92fac891e615515884d66efe09f','original_initialization_changed'
 atomic(ART/'initialization.json',{'native_head_sha256':initial_sha,'source':'unchanged native_model original VG asset, not any adapted best checkpoint','all_arms_share_initialization':True,'dropout_protocol':CFG['batching']['dropout']})
 event('new_matched_initialization_locked',native_head_sha256=initial_sha,test_scored=False)
 def run(phase,variant,seed,epochs):
  folder=ART/f'{phase}_{variant}_{seed}';folder.mkdir(exist_ok=True)
  if (folder/'report.json').exists():
   done=json.loads((folder/'report.json').read_text())
   if done.get('status')=='complete':return done
   raise RuntimeError('partial_run_requires_explicit_audited_resume')
  guard();verify();torch.manual_seed(seed);np.random.seed(seed);random.seed(seed);model.load_state_dict(initial,strict=False)
  adapt=Adaptation(model,variant).cuda();parameters=[p for p in adapt.parameters() if p.requires_grad]
  optimizer=torch.optim.AdamW(parameters,lr=CFG['optimizer']['lr'],weight_decay=CFG['optimizer']['weight_decay'])
  scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=max(1,epochs-CFG['optimizer']['drop_epoch_from_end']),gamma=.1)
  rng=random.Random(seed);history=[];best=None;fallback=None;start=time.time()
  report={'experiment_id':CFG['experiment_id'],'run':folder.name,'variant':variant,'seed':seed,'status':'running','selected_epoch':None,'history':history,'frozen_baseline_metrics':frozen,'test_scored':False,'microbatch':2,'initial_head_sha256':initial_sha}
  atomic(folder/'report.json',report)
  for epoch in range(1,epochs+1):
   guard();verify();t=time.time();adapt.train();optimizer.zero_grad(set_to_none=True);total=0.;steps=0
   torch.cuda.reset_peak_memory_stats()
   for group in grouped(items('train',rng),2):
    guard();targets=[target(train[item['index']]) for item in group];b=pack(group,pinned=True)
    pred,_=adapt(b);values=per_image_losses(criterion,pred,targets);finite_loss(values)
    (values.sum()/16).backward();steps+=len(group);total+=float(values.detach().sum())
    del b,pred,values,targets
    if step_boundary(steps,16):
     torch.nn.utils.clip_grad_norm_(parameters,CFG['optimizer']['gradient_clip'],error_if_nonfinite=True);optimizer.step();optimizer.zero_grad(set_to_none=True)
    if steps%256==0:event('training',run=folder.name,epoch=epoch,images=steps,total_images=sum(e['images'] for e in manifest['splits']['train']),loss=total/steps,step_seconds=(time.time()-t)/steps,peak_allocated_bytes=torch.cuda.max_memory_allocated(),**resources())
   if steps%16:
    torch.nn.utils.clip_grad_norm_(parameters,CFG['optimizer']['gradient_clip'],error_if_nonfinite=True);optimizer.step();optimizer.zero_grad(set_to_none=True)
   metrics=evaluate(adapt,'val',val,folder/'current_validation_records.json');entry={'epoch':epoch,'metrics':metrics,'train_loss':total/steps,'seconds':time.time()-t};history.append(entry)
   payload={'native_head':{k:v.detach().cpu() for k,v in model.state_dict().items() if 'add_add_add_relation' in k},'residual':None if adapt.residual is None else adapt.residual.state_dict(),'variant':variant,'seed':seed,'epoch':epoch,'experiment_id':CFG['experiment_id'],'microbatch':2}
   eligible=metrics['R@50']>=frozen['R@50']-CFG['screening']['max_R50_drop']
   if fallback is None or metrics['mR@50']>fallback['metrics']['mR@50']:
    fallback=entry;torch.save(payload,folder/'fallback.pt');shutil.copy2(folder/'current_validation_records.json',folder/'fallback_validation_records.json')
   if eligible and (best is None or metrics['mR@50']>best['metrics']['mR@50']):
    best=entry;torch.save(payload,folder/'best.pt');shutil.copy2(folder/'current_validation_records.json',folder/'best_validation_records.json')
   report.update(selected_epoch=(best or fallback)['epoch'],selected_metrics=(best or fallback)['metrics'],guardrail_met=best is not None);atomic(folder/'report.json',report)
   scheduler.step();artifact_guard();event('epoch_complete',run=folder.name,epoch=epoch,metrics=metrics,epoch_seconds=entry['seconds'],**resources())
  if best is None:shutil.copy2(folder/'fallback.pt',folder/'best.pt');shutil.copy2(folder/'fallback_validation_records.json',folder/'best_validation_records.json')
  report.update(status='complete',checkpoint=str((folder/'best.pt').relative_to(ROOT)),checkpoint_sha256=sha(folder/'best.pt'),wall_seconds=time.time()-start,completed_unix=time.time());atomic(folder/'report.json',report)
  (folder/'fallback.pt').unlink(missing_ok=True);event('checkpoint_complete',run=folder.name,selected_epoch=report['selected_epoch'],metrics=report['selected_metrics'])
  del adapt,parameters,optimizer,scheduler;model.zero_grad(set_to_none=True);import gc;gc.collect();torch.cuda.empty_cache()
  return report
 results={str(seed):run('replication','context',seed,13) for seed in [2718,3141]}
 inherited=json.loads((ROOT/'artifacts/vg_method_v2_speed_m2_v1/confirmation_context_1729/report.json').read_text())
 assert inherited['status']=='complete' and inherited['checkpoint_sha256']=='29017a69d7cb6d76d68c834b5d540712775a9e0cf9cae13efd17d78a7b69292e'
 all_rows={'1729':inherited,**results};summary={}
 for name in ['R@20','mR@20','R@50','mR@50','R@100','mR@100']:
  values=[row['selected_metrics'][name] for row in all_rows.values()]
  summary[name]={'by_seed':dict(zip(all_rows,values)),'mean':float(np.mean(values)),'sample_std':float(np.std(values,ddof=1))}
 result={'status':'CONTEXT_THREE_SEED_VALIDATION_COMPLETE','variant':'context','new_seeds':[2718,3141],'validation':summary,'runs':all_rows,'test_scored':False,'test_accessed':False,'claim':'Fixed-recipe validation replication only. Prior exploratory single-seed test is frozen; no regional superiority or fresh held-out test claim.'}
 atomic(ROOT/'reports/VG_CONTEXT_SEED_REPLICATION_V1_RESULT.json',result);event('stage_complete',outcome=result['status'],test_scored=False)
if __name__=='__main__':
 try:main()
 except Exception as e:
  out={'status':'stopped_on_error','error_type':type(e).__name__,'reason':str(e)[:800],'traceback':traceback.format_exc()[-5000:],'test_scored':False,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
  atomic(ROOT/'reports/VG_CONTEXT_SEED_REPLICATION_V1_RESULT.json',out);event('stopped_on_error',error_type=out['error_type'],reason=out['reason']);raise
