"""New four-arm matched microbatch2 experiment; cached-head adaptation only."""
from pathlib import Path
import os,sys,json,time,datetime,hashlib,subprocess,traceback,shutil,random,io,resource
ROOT=Path.home()/'rsg-former';ART=ROOT/'artifacts/vg_method_v2_speed_m2_v1'
CFG=json.loads((ROOT/'configs/vg_method_v2_time_extension_v1.json').read_text());START=time.time()
PREREG=json.loads((ART/'time_extension_v1/preregistration.json').read_text())
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
 atomic(ROOT/'reports/VG_V2_TIME_EXT_PROGRESS.json',o);print(json.dumps(o),flush=True)
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
 total=sum(p.stat().st_size for base in [ROOT/'artifacts/vg_method_v2',ART] for p in base.rglob('*') if p.is_file())
 assert total<=40*1024**3,'aggregate_artifact_budget_reached'

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
 test_manifest={'cache_dtype':'float32','splits':{'test':[]},'status':'building'}
 test_cache=ART/'features_test'
 def items(split,rng=None):
  entries=list((test_manifest if split=='test' else manifest)['splits'][split])
  if rng is not None:rng.shuffle(entries)
  ds=train if split=='train' else val if split=='val' else test
  for e in entries:
   guard();p=(test_cache if split=='test' else cache)/e['path'];raw=p.read_bytes()
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
    if (j+1)%128==0:event('validation' if split=='val' else 'locked_test',split=split,images=j+1,seconds_per_image=(time.time()-start)/(j+1),**resources())
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
 atomic(ART/'time_extension_v1/initialization.json',{'native_head_sha256':initial_sha,'source':'unchanged native_model original VG asset, not any adapted best checkpoint','all_arms_share_initialization':True,'dropout_protocol':CFG['batching']['dropout']})
 event('new_matched_initialization_locked',native_head_sha256=initial_sha,test_scored=False)
 def feature_cache(ds,split):
  # No train/validation recapture and no write to the original manifest.
  assert split=='test' and (ART/'test_lock.json').exists(),'test_cache_requires_registered_gate'
  test_cache.mkdir(exist_ok=False);chunk=[];ids=[]
  for i in range(len(ds)):
   guard();sample=ds[i];item,_=capture(model,ds.image(i));item={k:v.float() for k,v in item.items()};item.update(index=i,image_id=sample.image_id)
   chunk.append(item);ids.append(sample.image_id)
   if len(chunk)==64 or i==len(ds)-1:
    p=test_cache/f'test_{len(test_manifest["splits"]["test"]):05d}.pt';torch.save(chunk,p)
    test_manifest['splits']['test'].append({'path':p.name,'images':len(chunk),'image_ids':ids,'bytes':p.stat().st_size,'sha256':sha(p),'last_index':i})
    atomic(test_cache/'manifest.json',test_manifest);chunk=[];ids=[]
    current=sum(e['bytes'] for es in manifest['splits'].values() for e in es)+sum(e['bytes'] for e in test_manifest['splits']['test'])
    if current+PREREG['noncache_artifact_reserve_bytes']>40*1024**3:raise RuntimeError('aggregate_artifact_budget_reached')
    event('feature_cache',split='test',images=i+1,total_images=len(ds),**resources())
  test_manifest['status']='complete';atomic(test_cache/'manifest.json',test_manifest)
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
 screen={v:run('screening',v,CFG['screening']['seed'],CFG['screening']['epochs']) for v in CFG['screening']['variants']}
 def gain(a,b):return a['selected_metrics']['mR@50']-b['selected_metrics']['mR@50']
 candidates=[v for v in ['oneway','feedback'] if screen[v]['guardrail_met'] and all(gain(screen[v],screen[b])>=CFG['screening']['gate_gain_mR50'] and screen[v]['selected_metrics']['R@50']>=screen[b]['selected_metrics']['R@50']-.01 for b in ['native','context'])]
 if not candidates:
  result={'status':'screening_no_region_benefit','screening':screen,'best_validation_variant':max(screen,key=lambda v:screen[v]['selected_metrics']['mR@50']),'test_scored':False,'caption_stage':'pending separate clean-data detector training','claim':'No regional-method superiority established in this adaptation budget.'};atomic(ROOT/'reports/VG_V2_TIME_EXT_RESULT.json',result);event('sgdet_stage_finished',outcome=result['status']);return
 candidate=max(candidates,key=lambda v:screen[v]['selected_metrics']['mR@50'])
 variants=['native','context']+(['oneway','feedback'] if candidate=='feedback' else ['oneway'])
 # Conservative admission before spending confirmation or accessing test.
 epoch_bound=max(e['seconds'] for r in screen.values() for e in r['history'])*1.25
 planned=len(variants)*len(CFG['confirmation']['seeds'])*CFG['confirmation']['epochs']
 required=planned*epoch_bound+len(variants)*len(CFG['confirmation']['seeds'])*epoch_bound
 remaining=CFG['authorized_wall_deadline_unix']-time.time()
 atomic(ART/'confirmation_admission.json',{'remaining_seconds':remaining,'required_seconds':required,'planned_epochs':planned,'epoch_bound_seconds':epoch_bound,'admitted':required<=remaining,'scientific_gate_unchanged':True})
 if required>remaining:
  result={'experiment_id':CFG['experiment_id'],'status':'confirmation_resource_admission_blocked','screening':screen,'candidate':candidate,'test_scored':False,'caption_stage':'pending separate clean-data detector training','reason':'registered confirmation and test allowance do not fit authorized extended deadline; no gate weakening or partial confirmation'}
  atomic(ROOT/'reports/VG_V2_TIME_EXT_RESULT.json',result);event('sgdet_stage_finished',outcome=result['status']);return
 confirmation={str(seed):{v:run('confirmation',v,seed,CFG['confirmation']['epochs']) for v in variants} for seed in CFG['confirmation']['seeds']}
 # Paired image bootstrap of macro recall, seed-averaged. No independence
 # claim for three seeds: report each seed and all variance.
 def records(run):return json.loads((ROOT/Path(run['checkpoint']).parent/'best_validation_records.json').read_text())
 def bootstrap(control):
  aa=[records(confirmation[str(s)][candidate]) for s in CFG['confirmation']['seeds']];bb=[records(confirmation[str(s)][control]) for s in CFG['confirmation']['seeds']]
  n=len(aa[0]);a=np.full((3,n,50),np.nan);b=np.full_like(a,np.nan)
  for seed in range(3):
   assert [x['image_id'] for x in aa[seed]]==[x['image_id'] for x in bb[seed]]
   for i,(x,y) in enumerate(zip(aa[seed],bb[seed])):
    for p in x['predicates']:a[seed,i,int(p)-1]=x['predicates'][p]['50'];b[seed,i,int(p)-1]=y['predicates'][p]['50']
  rng=np.random.default_rng(CFG['confirmation']['bootstrap_seed']);values=[]
  for rep in range(CFG['confirmation']['bootstrap_replicates']):
   inds=rng.integers(n,size=n);delta=np.nanmean(a[:,inds,:],axis=1)-np.nanmean(b[:,inds,:],axis=1);values.append(float(np.nan_to_num(delta,nan=0).mean()))
  return {'lower95':float(np.quantile(values,.025)),'upper95':float(np.quantile(values,.975)),'unit':'absolute recall, multiply by100 for pp'}
 cis={b:bootstrap(b) for b in ['native','context']};valid=True;comparison={}
 for b in ['native','context']:
  deltas=[gain(confirmation[str(s)][candidate],confirmation[str(s)][b]) for s in CFG['confirmation']['seeds']]
  rd=[confirmation[str(s)][candidate]['selected_metrics']['R@50']-confirmation[str(s)][b]['selected_metrics']['R@50'] for s in CFG['confirmation']['seeds']]
  comparison[b]={'mR50_gain_by_seed':deltas,'R50_delta_by_seed':rd,'paired_ci':cis[b]}
  valid &= np.mean(deltas)>=.01 and min(rd)>=-.01 and cis[b]['lower95']>0
 if candidate=='feedback':
  valid &= np.mean([gain(confirmation[str(s)]['feedback'],confirmation[str(s)]['oneway']) for s in CFG['confirmation']['seeds']])>0
 result={'status':'confirmation_passed' if valid else 'confirmation_no_benefit','candidate':candidate,'screening':screen,'confirmation':confirmation,'comparisons':comparison,'test_scored':False,'caption_stage':'pending separate clean-data detector training'}
 atomic(ROOT/'reports/VG_V2_TIME_EXT_RESULT.json',result)
 if not valid:event('sgdet_stage_finished',outcome=result['status']);return
 # Locked test phase requires exact source/config locks and all gates above.
 assert all(sha(ROOT/p)==h for p,h in prereg['source_hashes'].items())
 test=datasets(ROOT,'test');test_reports={}
 atomic(ART/'test_lock.json',{'candidate':candidate,'models':confirmation,'test_split_hash':test.split_hash(),'preregistration':prereg,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
 feature_cache(test,'test')
 for s in CFG['confirmation']['seeds']:
  test_reports[str(s)]={}
  for v in variants:
   ckpt=torch.load(ROOT/confirmation[str(s)][v]['checkpoint'],map_location='cpu',weights_only=True);model.load_state_dict(ckpt['native_head'],strict=False)
   adapt=Adaptation(model,v).cuda()
   if adapt.residual:adapt.residual.load_state_dict(ckpt['residual'])
   metrics=evaluate(adapt,'test',test,ART/f'locked_test_{v}_{s}_records.json');test_reports[str(s)][v]=metrics
 result.update(status='locked_test_complete',test_scored=True,test=test_reports,claim='Report locked test comparisons before any superiority claim; caption evidence still pending.')
 atomic(ROOT/'reports/VG_V2_TIME_EXT_RESULT.json',result);event('sgdet_stage_finished',outcome=result['status'])
if __name__=='__main__':
 try:main()
 except Exception as e:
  out={'experiment_id':CFG['experiment_id'],'status':'stopped_on_error','error_type':type(e).__name__,'reason':str(e)[:800],'traceback':traceback.format_exc()[-5000:],'test_scored':(ART/'test_lock.json').exists(),'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
  atomic(ROOT/'reports/VG_V2_TIME_EXT_RESULT.json',out);event('stopped_on_error',error_type=out['error_type'],reason=out['reason']);raise
