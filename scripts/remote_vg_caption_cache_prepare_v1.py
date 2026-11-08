"""Immutable server TRAIN-bank/VAL SG cache and separate region references. No TEST."""
from pathlib import Path
from datetime import datetime,timezone
import gc,gzip,hashlib,io,json,os,resource,shutil,signal,sqlite3,time,zipfile
os.environ.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',CUDA_CACHE_DISABLE='1')
ROOT=Path.home()/'rsg-former';ART=ROOT/'artifacts/vg_caption_execution_v1_20261025';START=time.time();LAST_ARTIFACT_CHECK=0
os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:4]));signal.alarm(1800)
from rsgformer.vg_caption_graph_v1 import compact_graph,hash_order,focus_graph,digest
from rsgformer.vg_caption_region_units import prepare_provided_crop_units

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()

def write(path,value):
 tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)

def guard():
 global LAST_ARTIFACT_CHECK
 if time.time()-LAST_ARTIFACT_CHECK>15:
  bases=[ROOT/'artifacts'/name for name in ['vg_method_v2','vg_method_v2_speed_m2_v1','vg_component_control_test_v1','vg_caption_execution_v1_20261025']]
  assert sum(p.stat().st_size for base in bases for p in base.rglob('*') if p.is_file())<=40*1024**3,'aggregate_artifact40GiB_cap'
  if ART.exists():assert sum(p.stat().st_size for p in ART.rglob('*') if p.is_file())<=8*1024**3,'phase8GiB_cap'
  LAST_ARTIFACT_CHECK=time.time()
 assert time.time()<1794670321.2414355,'overall_authorized_deadline'
 assert shutil.disk_usage(ROOT).free>=17*1024**3,'disk_reserve_plus_phase_admission'
 assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=4*1024**3,'rss4GiB'
 assert not (ART/'STOP').exists(),'owned_stop'

def event(stage,**kw):
 guard();data={'utc':datetime.now(timezone.utc).isoformat(),'stage':stage,**kw}
 write(ART/'progress.json',data);print(json.dumps(data),flush=True)

def main():
 guard();assert not ART.exists(),'phase_already_exists_no_duplicate_or_hidden_resume'
 ART.mkdir();source={p:sha(ROOT/p) for p in ['scripts/remote_vg_caption_cache_prepare_v1.py','src/rsgformer/vg_caption_graph_v1.py','src/rsgformer/vg_object_color_v1.py','src/rsgformer/vg_caption_region_units.py']}
 prereg={'phase':'vg_caption_execution_v1_20261025','created_utc':datetime.now(timezone.utc).isoformat(),
  'human_authorization':'25Oct process cache on server then captions/multiple configurations; new Qwen36/32/12GiB CPU4 separately approved',
  'source_hashes':source,'checkpoint_sha256':'29017a69d7cb6d76d68c834b5d540712775a9e0cf9cae13efd17d78a7b69292e',
  'TRAIN_bank_images_cap':200,'VAL_caption_dev_images':64,'VAL_caption_check_images':64,'VAL_color_dev_images':200,'VAL_color_check_images':200,
  'regions_per_image_cap':4,'seed_training_allowed':False,'test_access_allowed':False,'new_sgdet_scoring':False,
  'namespace':'VG_CAPTION_EXEC_V1_20261025','cache_phase_cap_gib':8,'overall_artifact_cap_gib':40,'disk_reserve_gib':16,
  'cache_scope':'all5000VALbasegraphs;TRAIN200bankgraphs;TRAIN200/VAL528providedregionunits;candidate colors require separate VAL visual gate',
  'no_color_gt_in_model_inputs':True,'native_NMS_representative_rule':'max object score per equivalence group; keep query IDs/provenance; metrics untouched',
  'color_policies':'five extractors/three fixed global operating points in vg_object_color_v1; no TEST tuning',
  'qwen_test_phase_requires_new_prereg_and_color_visual_check':True}
 write(ART/'preregistration.json',prereg);event('source_and_membership_preparation')
 annotations={}
 with zipfile.ZipFile(ROOT/'data/vg_canonical_audit/reltr_vg_annotations.zip') as z:
  for split in ['train','val']:
   raw=z.read('vg/'+split+'.json');data=json.loads(raw)
   annotations[split]={int(x['id']):x for x in data['images']}
   if split=='val':objects={int(c['id']):c['name'] for c in data['categories'] if c['id']>0}
   del raw,data;gc.collect()
  predicates=dict(enumerate(json.loads(z.read('vg/rel.json'))['rel_categories']))
 assert len(annotations['train'])==57723 and len(annotations['val'])==5000
 val_order=sorted(annotations['val'],key=lambda i:hash_order(prereg['namespace']+'VAL',i))
 membership={'caption_dev':val_order[:64],'caption_check':val_order[64:128],
  'color_dev':val_order[128:328],'color_check':val_order[328:528],
  'train_bank':sorted(annotations['train'],key=lambda i:hash_order(prereg['namespace']+'TRAIN',i))[:200]}
 assert len(set(sum([membership[k] for k in ['caption_dev','caption_check','color_dev','color_check']],[])))==528
 manifest=json.loads((ROOT/'artifacts/vg_method_v2/features/manifest.json').read_text())
 cached={i for e in manifest['splits']['train'] for i in e['image_ids']}
 assert set(membership['train_bank'])<=cached,'selected_canonical_TRAIN_ID_missing_cache_no_resample'
 write(ART/'membership.json',membership)
 pred=ROOT/'artifacts/vg_component_control_test_v1/validation_predictions.jsonl.gz'
 assert sha(pred)=='99cf87a0f9248f4ba95e2f25c8e7127f5ff893a22841cf901683e49b47ff1378'
 graphs={};val_count=0
 with gzip.open(pred,'rt') as inp,gzip.open(ART/'val_base_sg.jsonl.gz','xt',encoding='utf-8') as out:
  for row_index,line in enumerate(inp):
   row=json.loads(line);identity=row['image_id'];meta=annotations['val'][identity]
   record={'image_id':identity,'split':'val','width':meta['width'],'height':meta['height'],
    'graph':compact_graph(row['prediction'],objects,predicates),
    'source_prediction_sha256':'99cf87a0f9248f4ba95e2f25c8e7127f5ff893a22841cf901683e49b47ff1378','source_row':row_index}
   record['base_graph_sha256']=digest(record['graph']);out.write(json.dumps(record)+'\n');val_count+=1
   if identity in set(val_order[:528]):graphs[identity]=record
   if val_count%1000==0:event('VAL_base_SG_caching',images=val_count)
 assert val_count==5000
 event('TRAIN_bank_SG_inference',images_cap=200)
 import torch,numpy as np
 from rsgformer.vg_v2_runtime import native_model,Adaptation
 from rsgformer.vg_native_postprocess import native_vg_postprocess
 torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(1729)
 torch.cuda.set_per_process_memory_fraction(6*1024**3/torch.cuda.get_device_properties(0).total_memory)
 assert torch.cuda.mem_get_info()[0]>=8*1024**3,'gpu8GiB_free_admission'
 checkpoint=ROOT/'artifacts/vg_method_v2_speed_m2_v1/confirmation_context_1729/best.pt'
 assert sha(checkpoint)==prereg['checkpoint_sha256']
 model,_,_=native_model(ROOT);ck=torch.load(checkpoint,map_location='cpu',weights_only=True)
 assert ck['seed']==1729 and ck['epoch']==10 and ck['variant']=='context'
 loaded=model.load_state_dict(ck['native_head'],strict=False);assert not loaded.unexpected_keys
 adapt=Adaptation(model,'context');adapt.residual.load_state_dict(ck['residual'],strict=True);adapt.cuda().eval();model.eval();del ck;gc.collect()
 wanted=set(membership['train_bank']);train_count=0
 with gzip.open(ART/'train_bank_base_sg.jsonl.gz','xt',encoding='utf-8') as out:
  for entry in manifest['splits']['train']:
   if not wanted&set(entry['image_ids']):continue
   path=ROOT/'artifacts/vg_method_v2/features'/entry['path'];raw=path.read_bytes()
   assert hashlib.sha256(raw).hexdigest()==entry['sha256'],'feature_shard_changed'
   chunk=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True);del raw
   for item in chunk:
    identity=item['image_id']
    if identity not in wanted:continue
    guard();meta=annotations['train'][identity]
    with torch.inference_mode():
     outputs,_=adapt({k:item[k][None].float().cuda() for k in ['nodes','boxes','logits','visual']})
     prediction=native_vg_postprocess(outputs,int(meta['width']),int(meta['height']))
    prediction={k:v.tolist() if hasattr(v,'tolist') else v for k,v in prediction.items()}
    record={'image_id':identity,'split':'train','width':meta['width'],'height':meta['height'],
     'graph':compact_graph(prediction,objects,predicates),'source_feature_shard_sha256':entry['sha256']}
    record['base_graph_sha256']=digest(record['graph']);out.write(json.dumps(record)+'\n');graphs[identity]=record;train_count+=1
   del chunk;gc.collect()
   if train_count and train_count%25==0:event('TRAIN_bank_SG_inference',images=train_count,allocator_bytes=torch.cuda.max_memory_allocated())
 assert train_count==200
 del adapt,model;gc.collect();torch.cuda.empty_cache()
 event('region_geometry_and_separate_references')
 expected=set(graphs);seen=set();refs={};units=[];rejected=[]
 decoder=json.JSONDecoder();buf=''
 with (ROOT/'data/vg_caption/region_descriptions.json').open(encoding='utf-8') as f:
  assert f.read(1)=='['
  while True:
   buf=buf.lstrip()
   if buf.startswith(','):buf=buf[1:].lstrip()
   if buf.startswith(']'):break
   try:item,end=decoder.raw_decode(buf)
   except json.JSONDecodeError:
    part=f.read(1024**2);assert part,'reference_source_truncated';buf+=part;continue
   buf=buf[end:];identity=item['id']
   if identity not in expected:continue
   record=graphs[identity];ins,rs,bad=prepare_provided_crop_units(identity,[record['width'],record['height']],item['regions'])
   selected=sorted(ins,key=lambda u:hash_order(prereg['namespace']+'REGION',u))[:4]
   for u in selected:
    unit=ins[u];unit['split']=record['split'];unit['partition']=next(k for k,ids in membership.items() if identity in ids)
    unit['graph']=focus_graph(record['graph'],unit['crop_xyxy']);unit['base_graph_sha256']=record['base_graph_sha256'];units.append(unit);refs[u]=rs[u]['human_references']
   rejected.extend(bad);seen.add(identity)
 assert seen==expected,'selected_image_missing_reference_record'
 with gzip.open(ART/'region_units_no_color.jsonl.gz','xt',encoding='utf-8') as out:
  for unit in units:out.write(json.dumps(unit)+'\n')
 write(ART/'human_references_separate.json',refs);write(ART/'rejected_region_records.json',rejected)
 complete={'status':'SERVER_BASE_SG_AND_REGION_CACHE_COMPLETE_TRAIN_VAL_ONLY','utc':datetime.now(timezone.utc).isoformat(),
  'train_graphs':train_count,'val_graphs':val_count,'region_units':len(units),'partition_units':{p:sum(u['partition']==p for u in units) for p in membership},
  'file_sha256':{p.name:sha(p) for p in ART.iterdir() if p.is_file() and p.name!='progress.json'},
  'no_evidence_units':sum(not u['graph']['nodes'] for u in units),'test_accessed':False,'caption_generated':False,
  'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'wall_seconds':time.time()-START}
 write(ART/'base_cache_complete.json',complete);event('BASE_CACHE_COMPLETE',**complete)

if __name__=='__main__':
 try:main()
 except Exception as e:
  if ART.exists():write(ART/'cache_failure.json',{'utc':datetime.now(timezone.utc).isoformat(),'error_type':type(e).__name__,'reason':str(e),'test_accessed':False})
  raise
