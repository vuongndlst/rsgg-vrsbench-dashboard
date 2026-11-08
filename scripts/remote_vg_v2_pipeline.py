"""Finite SGDet experiment; cache/prediction sources locked, tests gated.

No source edits, hidden retries, criterion changes, or test tuning are performed
inside this process. Caption-clean training remains an explicitly separate stage.
"""
from pathlib import Path
import os,sys,json,time,datetime,hashlib,subprocess,traceback,shutil,random
ROOT=Path.home()/'rsg-former';ART=ROOT/'artifacts/vg_method_v2';ART.mkdir(parents=True,exist_ok=True)
CFG=json.loads((ROOT/'configs/vg_method_v2.json').read_text());START=time.time()
def atomic(path,value):
 t=path.with_suffix(path.suffix+'.tmp');t.write_text(json.dumps(value,indent=2));t.replace(path)
def event(stage,**kw):
 o={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'stage':stage,'wall_seconds':time.time()-START,**kw}
 atomic(ROOT/'reports/VG_V2_PROGRESS.json',o);print(json.dumps(o),flush=True)
def guard():
 if (ART/'STOP').exists():raise RuntimeError('user_stop_flag')
 if time.time()-START>CFG['resource']['wall_hours']*3600:raise RuntimeError('wall_budget_reached')
 if shutil.disk_usage(ROOT).free<CFG['resource']['minimum_free_disk_gib']*1024**3:raise RuntimeError('disk_reserve_reached')
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for part in iter(lambda:f.read(8*1024*1024),b''):h.update(part)
 return h.hexdigest()

def main():
 # The prepare process is independent; waiting is bounded and carries actual
 # downloaded bytes so a monitor can distinguish progress from a stalled job.
 deadline=time.time()+CFG['resource']['preparation_wait_minutes']*60
 while not (ROOT/'reports/VG_V2_ASSETS.json').exists():
  guard();p=ROOT/'reports/VG_V2_PROGRESS.json'
  if p.exists() and json.loads(p.read_text()).get('stage')=='preparation_failed':raise RuntimeError('asset_preparation_failed')
  if time.time()>deadline:raise RuntimeError('asset_preparation_timeout')
  files=list((ROOT/'vendor/salience_sgg/logs').glob('*'))
  event('awaiting_vg_assets',download_bytes=sum(p.stat().st_size for p in files if p.is_file()));time.sleep(30)
 probe=subprocess.run([sys.executable,str(ROOT/'scripts/remote_vg_v2_probe.py')],cwd=ROOT,capture_output=True,text=True,timeout=600)
 (ART/'probe.log').write_text(probe.stdout+'\n'+probe.stderr)
 if probe.returncode:raise RuntimeError('train_only_probe_failed: see VG_V2_PROBE.json')
 checked=json.loads((ROOT/'reports/VG_V2_PROBE.json').read_text());assert checked['status']=='PASS'
 import torch,numpy as np
 from rsgformer.vg_v2_runtime import native_model,datasets,capture,Adaptation,SGDetEvaluator,predicted_graph
 torch.set_num_threads(CFG['resource']['cpu_threads']);torch.set_float32_matmul_precision('highest')
 torch.cuda.set_per_process_memory_fraction(CFG['resource']['cuda_allocator_gib']*1024**3/torch.cuda.get_device_properties(0).total_memory)
 model,criterion,_=native_model(ROOT);model.cuda().eval()
 train=datasets(ROOT,'train');val=datasets(ROOT,'val')
 prereg={'config':CFG,'source_hashes':{p:sha(ROOT/p) for p in ['src/rsgformer/vg_v2_runtime.py','scripts/remote_vg_v2_pipeline.py','scripts/remote_vg_v2_probe.py','configs/vg_method_v2.json']},'upstream_commit':CFG['upstream_commit'],'train_images':len(train),'validation_images':len(val),'split_hashes':{'train':train.split_hash(),'val':val.split_hash()},'assets':json.loads((ROOT/'reports/VG_V2_ASSETS.json').read_text()),'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'test_scored':False,'cache_dtype':checked['cache_dtype']}
 atomic(ROOT/'reports/VG_V2_PREREGISTRATION.json',prereg)
 atomic(ART/'preregistration.json',prereg)
 cache=ART/'features';cache.mkdir(exist_ok=True)
 manifest={'cache_dtype':checked['cache_dtype'],'preregistration_source_hashes':prereg['source_hashes'],'splits':{},'status':'building'}
 if (cache/'manifest.json').exists():
  previous=json.loads((cache/'manifest.json').read_text());assert previous['preregistration_source_hashes']==manifest['preregistration_source_hashes'];manifest=previous
 def feature_cache(ds,split):
  entries=manifest['splits'].setdefault(split,[]);done=(entries[-1]['last_index']+1) if entries else 0;start=time.time()
  done_images=sum(e['images'] for e in entries)
  chunk=[];ids=[];excluded=[]
  for i in range(done,len(ds)):
   guard();sample=ds[i]
   if split=='train' and (len(sample.labels)>200 or len(sample.labels)==0 or len(sample.relations)==0):excluded.append(sample.image_id);continue
   item,_=capture(model,ds.image(i));dtype=torch.float16 if checked['cache_dtype']=='float16' else torch.float32
   item={k:v.to(dtype) for k,v in item.items()};item['index']=i;item['image_id']=sample.image_id
   chunk.append(item);ids.append(sample.image_id)
   if len(chunk)==CFG['cache']['shard_images'] or i==len(ds)-1:
    p=cache/f'{split}_{len(entries):05d}.pt';torch.save(chunk,p);entries.append({'path':p.name,'images':len(chunk),'image_ids':ids,'bytes':p.stat().st_size,'sha256':sha(p),'last_index':i})
    atomic(cache/'manifest.json',manifest);chunk=[];ids=[]
    bytes_=sum(e['bytes'] for es in manifest['splits'].values() for e in es)
    if bytes_>CFG['resource']['artifact_limit_gib']*1024**3*.8:raise RuntimeError('feature_artifact_budget_reached')
    finished=sum(e['images'] for e in entries);elapsed=time.time()-start;rate=max(.001,(finished-done_images)/elapsed)
    event('feature_cache',split=split,images=finished,total_images=len(ds),cache_bytes=bytes_,seconds_per_image=1/rate,estimated_remaining_seconds=(len(ds)-i-1)/rate)
  if chunk:
   p=cache/f'{split}_{len(entries):05d}.pt';torch.save(chunk,p);entries.append({'path':p.name,'images':len(chunk),'image_ids':ids,'bytes':p.stat().st_size,'sha256':sha(p),'last_index':len(ds)-1})
  manifest.setdefault('excluded_train_image_ids',[]).extend(excluded);atomic(cache/'manifest.json',manifest)
 def items(split,rng=None):
  entries=list(manifest['splits'][split]);
  if rng is not None:rng.shuffle(entries)
  for e in entries:
   chunk=torch.load(cache/e['path'],map_location='cpu',weights_only=True)
   if rng is not None:rng.shuffle(chunk)
   for item in chunk:yield item
 def batch(item):return {k:item[k][None].float().cuda() for k in ['nodes','boxes','logits','visual']}
 def evaluate(adapt,split,ds,records_path=None):
  adapt.eval();model.eval();ev=SGDetEvaluator();start=time.time()
  with torch.no_grad():
   for j,item in enumerate(items(split)):
    guard();sample=ds[item['index']];pred,_=adapt(batch(item));ev.update(sample,*predicted_graph(pred,sample))
    if (j+1)%128==0:event('validation' if split=='val' else 'locked_test',split=split,images=j+1,seconds_per_image=(time.time()-start)/(j+1))
  if records_path:atomic(records_path,ev.records)
  return ev.result()
 feature_cache(val,'val')
 base=Adaptation(model,'native').cuda();frozen=evaluate(base,'val',val,ART/'frozen_native_validation_records.json')
 atomic(ART/'frozen_native_validation.json',frozen);event('frozen_baseline_complete',metrics=frozen)
 feature_cache(train,'train');manifest['status']='complete';atomic(cache/'manifest.json',manifest)
 # Snapshot initial head state on CPU; every arm starts identically.
 initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items() if 'add_add_add_relation' in k}
 def run(phase,variant,seed,epochs):
  folder=ART/f'{phase}_{variant}_{seed}';folder.mkdir(exist_ok=True)
  if (folder/'report.json').exists():
   done=json.loads((folder/'report.json').read_text())
   if done.get('status')=='complete':return done
   raise RuntimeError('partial_run_requires_explicit_audited_resume')
  guard();torch.manual_seed(seed);np.random.seed(seed);random.seed(seed);model.load_state_dict(initial,strict=False)
  adapt=Adaptation(model,variant).cuda();parameters=[p for p in adapt.parameters() if p.requires_grad]
  optimizer=torch.optim.AdamW(parameters,lr=CFG['optimizer']['lr'],weight_decay=CFG['optimizer']['weight_decay'])
  scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=max(1,epochs-CFG['optimizer']['drop_epoch_from_end']),gamma=.1)
  rng=random.Random(seed);history=[];best=None;fallback=None;start=time.time()
  report={'run':folder.name,'variant':variant,'seed':seed,'status':'running','selected_epoch':None,'history':history,'frozen_baseline_metrics':frozen,'test_scored':False}
  for epoch in range(1,epochs+1):
   t=time.time();adapt.train();optimizer.zero_grad(set_to_none=True);total=0.;steps=0
   for item in items('train',rng):
    guard();sample=train[item['index']];b=batch(item);box=torch.tensor(sample.boxes,device='cuda')/torch.tensor([sample.width,sample.height,sample.width,sample.height],device='cuda')
    rel=torch.zeros(200,200,50,device='cuda');indices=torch.tensor(sample.relations,device='cuda');rel[indices[:,0],indices[:,1],indices[:,2]-1]=1
    target={'labels':torch.tensor(sample.labels-1,device='cuda'),'boxes':torch.cat([(box[:,:2]+box[:,2:])/2,box[:,2:]-box[:,:2]],-1),'rel':rel}
    pred,_=adapt(b);losses=criterion(pred,[target]);loss=sum(losses[k]*criterion.weight_dict[k] for k in losses if k in criterion.weight_dict)
    if not torch.isfinite(loss):raise RuntimeError('non_finite_loss')
    (loss/CFG['optimizer']['effective_batch']).backward();steps+=1;total+=float(loss.detach())
    if steps%CFG['optimizer']['effective_batch']==0:
     torch.nn.utils.clip_grad_norm_(parameters,CFG['optimizer']['gradient_clip']);optimizer.step();optimizer.zero_grad(set_to_none=True)
    if steps%256==0:event('training',run=folder.name,epoch=epoch,images=steps,total_images=sum(e['images'] for e in manifest['splits']['train']),loss=total/steps,step_seconds=(time.time()-t)/steps,peak_allocated_bytes=torch.cuda.max_memory_allocated())
   if steps%CFG['optimizer']['effective_batch']:
    torch.nn.utils.clip_grad_norm_(parameters,CFG['optimizer']['gradient_clip']);optimizer.step();optimizer.zero_grad(set_to_none=True)
   metrics=evaluate(adapt,'val',val,folder/'current_validation_records.json');entry={'epoch':epoch,'metrics':metrics,'train_loss':total/steps,'seconds':time.time()-t};history.append(entry)
   payload={'native_head':{k:v.detach().cpu() for k,v in model.state_dict().items() if 'add_add_add_relation' in k},'residual':None if adapt.residual is None else adapt.residual.state_dict(),'variant':variant,'seed':seed,'epoch':epoch}
   eligible=metrics['R@50']>=frozen['R@50']-CFG['screening']['max_R50_drop']
   if fallback is None or metrics['mR@50']>fallback['metrics']['mR@50']:
    fallback=entry;torch.save(payload,folder/'fallback.pt');shutil.copy2(folder/'current_validation_records.json',folder/'fallback_validation_records.json')
   if eligible and (best is None or metrics['mR@50']>best['metrics']['mR@50']):
    best=entry;torch.save(payload,folder/'best.pt');shutil.copy2(folder/'current_validation_records.json',folder/'best_validation_records.json')
   report.update(selected_epoch=(best or fallback)['epoch'],selected_metrics=(best or fallback)['metrics'],guardrail_met=best is not None);atomic(folder/'report.json',report)
   scheduler.step();event('epoch_complete',run=folder.name,epoch=epoch,metrics=metrics,epoch_seconds=entry['seconds'])
  if best is None:shutil.copy2(folder/'fallback.pt',folder/'best.pt');shutil.copy2(folder/'fallback_validation_records.json',folder/'best_validation_records.json')
  report.update(status='complete',checkpoint=str((folder/'best.pt').relative_to(ROOT)),checkpoint_sha256=sha(folder/'best.pt'),wall_seconds=time.time()-start,completed_unix=time.time());atomic(folder/'report.json',report)
  # Keep one chosen checkpoint plus per-epoch metrics, rather than optimizer/backups.
  (folder/'fallback.pt').unlink(missing_ok=True);event('checkpoint_complete',run=folder.name,selected_epoch=report['selected_epoch'],metrics=report['selected_metrics']);return report
 screen={v:run('screening',v,CFG['screening']['seed'],CFG['screening']['epochs']) for v in CFG['screening']['variants']}
 def gain(a,b):return a['selected_metrics']['mR@50']-b['selected_metrics']['mR@50']
 candidates=[v for v in ['oneway','feedback'] if screen[v]['guardrail_met'] and all(gain(screen[v],screen[b])>=CFG['screening']['gate_gain_mR50'] and screen[v]['selected_metrics']['R@50']>=screen[b]['selected_metrics']['R@50']-.01 for b in ['native','context'])]
 if not candidates:
  result={'status':'screening_no_region_benefit','screening':screen,'best_validation_variant':max(screen,key=lambda v:screen[v]['selected_metrics']['mR@50']),'test_scored':False,'caption_stage':'pending separate clean-data detector training','claim':'No regional-method superiority established in this adaptation budget.'};atomic(ROOT/'reports/VG_V2_RESULT.json',result);event('sgdet_stage_finished',outcome=result['status']);return
 candidate=max(candidates,key=lambda v:screen[v]['selected_metrics']['mR@50'])
 variants=['native','context']+(['oneway','feedback'] if candidate=='feedback' else ['oneway'])
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
 atomic(ROOT/'reports/VG_V2_RESULT.json',result)
 if not valid:event('sgdet_stage_finished',outcome=result['status']);return
 # Locked test phase requires exact source/config locks and all gates above.
 assert all(sha(ROOT/p)==h for p,h in prereg['source_hashes'].items())
 test=datasets(ROOT,'test');feature_cache(test,'test');test_reports={}
 atomic(ART/'test_lock.json',{'candidate':candidate,'models':confirmation,'test_split_hash':test.split_hash(),'preregistration':prereg,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
 for s in CFG['confirmation']['seeds']:
  test_reports[str(s)]={}
  for v in variants:
   ckpt=torch.load(ROOT/confirmation[str(s)][v]['checkpoint'],map_location='cpu',weights_only=True);model.load_state_dict(ckpt['native_head'],strict=False)
   adapt=Adaptation(model,v).cuda()
   if adapt.residual:adapt.residual.load_state_dict(ckpt['residual'])
   metrics=evaluate(adapt,'test',test,ART/f'locked_test_{v}_{s}_records.json');test_reports[str(s)][v]=metrics
 result.update(status='locked_test_complete',test_scored=True,test=test_reports,claim='Report locked test comparisons before any superiority claim; caption evidence still pending.')
 atomic(ROOT/'reports/VG_V2_RESULT.json',result);event('sgdet_stage_finished',outcome=result['status'])
if __name__=='__main__':
 try:main()
 except Exception as e:
  out={'status':'stopped_on_error','error_type':type(e).__name__,'reason':str(e)[:800],'traceback':traceback.format_exc()[-5000:],'test_scored':(ART/'test_lock.json').exists(),'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
  atomic(ROOT/'reports/VG_V2_RESULT.json',out);event('stopped_on_error',error_type=out['error_type'],reason=out['reason']);raise
