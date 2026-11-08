"""Finite paired TRAIN benchmark: original masks/rules, no TEST or check tuning."""
from pathlib import Path
from datetime import datetime,timezone
import collections,gc,gzip,hashlib,io,json,math,os,resource,shutil,signal,sqlite3,statistics,subprocess,sys,time,traceback,zipfile
os.environ.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUDA_CACHE_DISABLE='1',TOKENIZERS_PARALLELISM='false')
ROOT=Path.home()/'rsg-former';BASE=ROOT/'artifacts/vg_caption_execution_v1_20261025';ART=ROOT/'artifacts/vg_color_speed_benchmark_v1_20261025';START=time.monotonic()
os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:4]));signal.alarm(900)
sys.path.insert(0,str(ROOT/'third_party/sam2_color_audit_20261025'))
import numpy as np,torch,cv2
from PIL import Image
from rsgformer.vg_object_color_rules_v3 import ColorNames
from rsgformer.vg_color_fast_engine_v1 import ColorEngine,MemoColorNames
from rsgformer.vg_object_color_v2 import ClipVerifier
from rsgformer.vg_color_binding_native_v1 import NeutralBinding
from rsgformer.numpy_json_adapter_v1 import numpy_json_default
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,indent=2,default=numpy_json_default,allow_nan=False));t.replace(p)
def guard():
 assert time.time()<1794670321.2414355 and time.monotonic()-START<900,'wall_cap'
 assert shutil.disk_usage(ROOT).free>=16*1024**3,'disk_reserve16GiB'
 assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=4*1024**3,'RSS4GiB'
 assert not (ART/'STOP').exists(),'owned_STOP'
 q=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
 used=sum(int(x.split(',')[1])*1024**2 for x in q.splitlines() if int(x.split(',')[0])==os.getpid())
 assert used<=8*1024**3,'own_GPU8GiB'
def compare(a,b,tol,path=''):
 diffs=[];maximum=0.
 if isinstance(a,dict) and isinstance(b,dict):
  if set(a)!=set(b):return [path+':keys'],maximum
  for key in a:
   d,m=compare(a[key],b[key],tol,path+'/'+key);diffs+=d;maximum=max(maximum,m)
 elif isinstance(a,(list,tuple)) and isinstance(b,(list,tuple)):
  if len(a)!=len(b):return [path+':length'],maximum
  for i,(x,y) in enumerate(zip(a,b)):
   d,m=compare(x,y,tol,path+'/'+str(i));diffs+=d;maximum=max(maximum,m)
 elif isinstance(a,(float,np.floating)) or isinstance(b,(float,np.floating)):
  if a is None or b is None:return ([path] if a!=b else []),maximum
  maximum=abs(float(a)-float(b))
  if not math.isfinite(maximum) or maximum>tol:diffs.append(path)
 elif a!=b:diffs.append(path)
 return diffs,maximum
def main():
 assert not ART.exists();guard();ART.mkdir()
 config_path=ROOT/'configs/vg_color_speed_benchmark_v1_20261025.json';cfg=json.loads(config_path.read_text())
 train=ROOT/'artifacts/color_rules_v4_train_pilot_20261025';assert (train/'complete.json').exists()
 targets=json.loads((BASE/'color_candidates_v2/target_node_manifest.json').read_text());targets=[t for t in targets if t['partition']=='train_bank'];assert len(targets)==200
 order=sorted({t['image_id'] for t in targets},key=lambda i:hashlib.sha256(('COLOR_SPEED_TRAIN_V1|'+str(i)).encode()).hexdigest())[:32]
 selected=[]
 for i in order:
  local=sorted([t for t in targets if t['image_id']==i],key=lambda t:hashlib.sha256(('COLOR_SPEED_NODE_V1|'+t['key']).encode()).hexdigest())[:3]
  selected+=local
 assert len(selected)<=96 and len(order)==32
 save(ART/'membership.json',selected)
 sources=['scripts/remote_vg_color_speed_benchmark_v1_20261025.py','src/rsgformer/vg_color_fast_engine_v1.py','src/rsgformer/vg_object_color_rules_v3.py',
  'src/rsgformer/vg_object_color_v2.py','src/rsgformer/vg_color_binding_native_v1.py','src/rsgformer/sam_binary_mask_adapter_v1.py',
  'configs/vg_color_speed_benchmark_v1_20261025.json','configs/vg_object_color_rules_v3_20261025.json','configs/vg_color_object_binding_native_v1_20261025.json']
 lock={'created_utc':datetime.now(timezone.utc).isoformat(),'phase':cfg['phase'],'config':cfg,'source_sha256':{p:sha(ROOT/p) for p in sources},
  'membership_sha256':sha(ART/'membership.json'),'parent_TRAIN_records_sha256':sha(train/'candidate_records.jsonl.gz'),'TEST_access_allowed':False,'VAL_check_reopened':False}
 save(ART/'preregistration.json',lock)
 with gzip.open(train/'candidate_records.jsonl.gz','rt') as f:historical={r['key']:r for r in map(json.loads,f)}
 with gzip.open(BASE/'train_bank_base_sg.jsonl.gz','rt') as f:graphs={r['image_id']:r['graph'] for r in map(json.loads,f) if r['image_id'] in set(order)}
 rules=json.loads((ROOT/'configs/vg_object_color_rules_v3_20261025.json').read_text());binding_cfg=json.loads((ROOT/'configs/vg_color_object_binding_native_v1_20261025.json').read_text())
 torch.set_num_threads(4);torch.set_num_interop_threads(1);cv2.setNumThreads(1)
 torch.cuda.set_per_process_memory_fraction(6*1024**3/torch.cuda.get_device_properties(0).total_memory)
 assert torch.cuda.mem_get_info()[0]>=8*1024**3
 weights=Path.home()/'etechs/thesis/var/models/sam2/sam2.1_hiera_small.pt';assert sha(weights)=='6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38'
 from sam2.build_sam import build_sam2
 from sam2.sam2_image_predictor import SAM2ImagePredictor
 predictor=SAM2ImagePredictor(build_sam2('configs/sam2.1/sam2.1_hiera_s.yaml',str(weights),apply_postprocessing=False),max_hole_area=0,max_sprinkle_area=0)
 clip=ClipVerifier(Path.home()/'shared-assets/models/clip-vit-base-patch32');binding=NeutralBinding(clip,rules['class_rules'],binding_cfg)
 table=ROOT/'artifacts/color_research_v3_20261025/w2c.mat';plain=ColorNames(table);memo=MemoColorNames(table)
 dummy=Image.new('RGB',(64,64),(127,127,127))
 for label in sorted({t['node']['label'] for t in selected}):clip.scores([dummy,dummy],label)
 archives=[zipfile.ZipFile(Path.home()/'sbir/hienlth/vg'/n) for n in ['images.zip','images2.zip']];members={}
 for z in archives:
  for m in z.infolist():
   if not m.filename.lower().endswith('.jpg'):continue
   try:i=int(Path(m.filename).stem)
   except ValueError:continue
   if i in order:assert i not in members;members[i]=(z,m)
 assert set(members)==set(order)
 with sqlite3.connect((ROOT/'artifacts/caption_readiness/vg_pixel_fingerprints_v1.sqlite').as_uri()+'?mode=ro',uri=True) as db:
  pixels={i:(h,w,hh) for i,h,w,hh in db.execute('SELECT vg_id,jpeg_sha,width,height FROM pixels') if i in order}
 # Common untimed SAM/CLIP warmup, same frozen TRAIN first image.
 z,m=members[order[0]];blob=z.read(m);image=Image.open(io.BytesIO(blob)).convert('RGB')
 warm=ColorEngine(predictor,clip,binding,plain,rules,'baseline');warm.set_image(image,order[0]);warm.run(selected[0],graphs[order[0]]);torch.cuda.synchronize();gc.collect()
 trials=[];outputs={}
 for repeat,sequence in enumerate(cfg['repeat_orders']):
  for mode in sequence:
   guard();engine=ColorEngine(predictor,clip,binding,plain if mode=='baseline' else memo,rules,mode);current=None;rows={}
   gc.collect();torch.cuda.synchronize();start=time.perf_counter()
   with gzip.open(ART/(str(repeat)+'_'+mode+'.jsonl.gz'),'xt') as out:
    for index,target in enumerate(selected):
     guard();identity=target['image_id']
     if identity!=current:
      z,m=members[identity];raw=z.read(m);assert hashlib.sha256(raw).hexdigest()==pixels[identity][0]
      image=Image.open(io.BytesIO(raw)).convert('RGB');assert image.size==pixels[identity][1:];engine.set_image(image,identity);current=identity
     row=engine.run(target,graphs[identity]);rows[target['key']]=json.loads(json.dumps(row,default=numpy_json_default,allow_nan=False))
     out.write(json.dumps(row,default=numpy_json_default,allow_nan=False)+'\n');out.flush()
     if mode=='baseline' or (index+1)%16==0:gc.collect()
   torch.cuda.synchronize();seconds=time.perf_counter()-start
   trials.append({'repeat':repeat,'mode':mode,'nodes':len(selected),'images':len(order),'seconds':seconds,'nodes_per_second':len(selected)/seconds,'component_seconds':engine.times,'counters':engine.counters})
   outputs[(repeat,mode)]=rows;save(ART/'progress.json',{'utc':datetime.now(timezone.utc).isoformat(),'completed_trials':len(trials),'planned_trials':6,'last_trial':trials[-1],'TEST_accessed':False})
   assert sum(p.stat().st_size for p in ART.rglob('*') if p.is_file())<=128*1024**2
   gc.collect()
 for z in archives:z.close()
 parity={};historical_mismatches=[]
 for key,row in outputs[(0,'baseline')].items():
  for field in ['methods','native_object_binding','CLIP_fill127_fill255','excluded_child_node_ids']:
   diffs,_=compare(row[field],historical[key][field],1e-6)
   if diffs:historical_mismatches.append({'key':key,'field':field,'paths':diffs[:5]})
 for mode in ['safe_cache','shared_clip']:
  discrepancies=[];maximum=0
  for repeat in [0,1]:
   for key,reference in outputs[(repeat,'baseline')].items():
    d,m=compare(reference,outputs[(repeat,mode)][key],1e-8 if mode=='safe_cache' else 1e-6)
    if d:discrepancies.append({'repeat':repeat,'key':key,'paths':d[:6]})
    maximum=max(maximum,m)
  speedups=[next(t['seconds'] for t in trials if t['repeat']==r and t['mode']=='baseline')/next(t['seconds'] for t in trials if t['repeat']==r and t['mode']==mode) for r in [0,1]]
  parity[mode]={'parity_pass':not discrepancies,'differences':discrepancies[:12],'difference_cases':len(discrepancies),'maximum_numeric_difference':maximum,
   'paired_measured_speedups':speedups,'minimum_speedup':min(speedups),'adoption_ready_engineering_only':not discrepancies and not historical_mismatches and min(speedups)>=1.10}
 result={'status':'FINITE_TRAIN_COLOR_SPEED_BENCHMARK_COMPLETE','utc':datetime.now(timezone.utc).isoformat(),'images':len(order),'nodes_per_trial':len(selected),'trials':trials,
  'parity':parity,'historical_TRAIN_reference_reproduced':not historical_mismatches,'historical_mismatches':historical_mismatches[:12],
  'peak_allocator_bytes':torch.cuda.max_memory_allocated(),'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'wall_seconds':time.monotonic()-START,
  'TEST_accessed':False,'VAL_check_reopened':False,'human_quality_gate_passed':False,'full_TEST_launch_allowed':False,
  'interpretation':'Engineering parity measures consistency, not real visual accuracy;160heldout review instances still need independent human labels'}
 save(ART/'complete.json',result);save(ROOT/'reports/VG_COLOR_SPEED_BENCHMARK_V1_20261025.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':
 try:main()
 except Exception as error:
  if ART.exists():save(ART/'failure.json',{'error_type':type(error).__name__,'reason':str(error),'TEST_accessed':False})
  traceback.print_exc();raise
