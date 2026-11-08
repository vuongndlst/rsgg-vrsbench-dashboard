"""Freeze100VAL images/one provided region each, then cache unchanged candidate colors."""
from pathlib import Path
import collections,gc,gzip,json,os,signal,time,traceback
os.environ.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUDA_CACHE_DISABLE='1')
ROOT=Path.home()/'rsg-former';BASE=ROOT/'artifacts/vg_caption_execution_v1_20261025';PIPE=ROOT/'artifacts/vg_caption_pilot100_v1_20261026';ART=PIPE/'prepare'
os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:4]));signal.alarm(3600)
from rsgformer.vg_color_execution_common_v1 import Guard,ImageZIPs,load_models,sha,save,line,now
from rsgformer.vg_color_full_test_policy_v1 import select_nodes
from rsgformer.vg_caption_graph_v1 import hash_order,digest

def main():
 assert not ART.exists();guard=Guard(ROOT,PIPE,3600,1);guard.check(False);ART.mkdir()
 lock=json.loads((PIPE/'preregistration.json').read_text())
 for p,h in lock['source_sha256'].items():assert sha(ROOT/p)==h
 done=json.loads((BASE/'base_cache_complete.json').read_text())
 for name in ['membership.json','region_units_no_color.jsonl.gz','train_bank_base_sg.jsonl.gz','val_base_sg.jsonl.gz']:
  assert sha(BASE/name)==done['file_sha256'][name]
 membership=json.loads((BASE/'membership.json').read_text());allowed=set(membership['caption_dev']+membership['color_dev']);excluded=set(membership['caption_check']+membership['color_check'])
 assert not allowed&excluded
 with gzip.open(BASE/'region_units_no_color.jsonl.gz','rt') as f:units=list(map(json.loads,f))
 by_image=collections.defaultdict(list)
 for u in units:
  if u['split']=='val' and u['parent_image_id'] in allowed:by_image[u['parent_image_id']].append(u)
 ids=sorted(by_image,key=lambda i:hash_order('VG_CAPTION_PILOT100_V1_20261026_IMAGE',i))[:100];assert len(ids)==100
 pilot=[min(by_image[i],key=lambda u:hash_order('VG_CAPTION_PILOT100_V1_20261026_REGION',u['unit_id'])) for i in ids]
 train=[u for u in units if u['partition']=='train_bank'];train_ids=set(membership['train_bank']);assert len(train_ids)==200
 wanted=set(ids)|train_ids;graphs={}
 for name in ['train_bank_base_sg.jsonl.gz','val_base_sg.jsonl.gz']:
  with gzip.open(BASE/name,'rt') as f:
   for r in map(json.loads,f):
    if r['image_id'] in wanted:assert digest(r['graph'])==r['base_graph_sha256'];graphs[r['image_id']]=r['graph']
 assert set(graphs)==wanted and len(wanted)==300
 cfg=json.loads((ROOT/'configs/vg_color_automated_full_test_v1_20261025.json').read_text());targets=[]
 for identity in sorted(wanted):
  for n in select_nodes(graphs[identity],set(cfg['eligible_classes']),4):targets.append({'key':str(identity)+':'+n['id'],'image_id':identity,'node':n})
 assert len(targets)<=1200
 save(ART/'membership.json',{'pilot_VAL_images':ids,'TRAIN_bank_images':sorted(train_ids),'excluded_caption_color_check_images':sorted(excluded),'pilot_unit_ids':[u['unit_id'] for u in pilot]})
 save(ART/'preregistration.json',{'utc':now(),'parent_lock_sha256':sha(PIPE/'preregistration.json'),'base_cache_completion_sha256':sha(BASE/'base_cache_complete.json'),
  'targets':len(targets),'pilot_images':100,'pilot_units':100,'TRAIN_bank_images':200,'selection_reference_and_color_blind':True,'TEST_access_allowed':False,'human_precision_gate_passed':False})
 save(ART/'target_manifest.json',targets)
 images=ImageZIPs(ROOT,wanted);engine,torch=load_models(ROOT,guard);current=None;colors={};count=0
 with gzip.open(ART/'candidate_records.jsonl.gz','xt',compresslevel=3) as f:
  for t in targets:
   guard.check();identity=t['image_id']
   if identity!=current:engine.set_image(images.read(identity),identity);current=identity
   row=engine.run(t,graphs[identity]);row.update(image_id=identity,node=t['node'],**{'class':t['node']['label']});f.write(line(row)+'\n')
   method=row['methods']['SAM2_ColorNames'];binding=bool(row.get('native_object_binding') and row['native_object_binding']['profile_pass']['strict'])
   colors[t['key']]=method.get('operating_points',{}).get('conservative',{}).get('colors',[]) if binding else []
   count+=1
   if count%16==0:gc.collect()
   if count%100==0:f.flush();save(ART/'progress.json',{'utc':now(),'nodes':count,'expected':len(targets),'TEST_accessed':False})
 images.close()
 for name,selected in [('pilot_units.jsonl.gz',pilot),('train_units.jsonl.gz',train)]:
  with gzip.open(ART/name,'xt') as f:
   for u in selected:
    for n in u['graph']['nodes']:n['candidate_colors']=colors.get(str(u['parent_image_id'])+':'+n['id'],[])
    f.write(line(u)+'\n')
 result={'status':'PILOT100_TRAIN_VAL_CANDIDATE_CACHE_COMPLETE','utc':now(),'pilot_images':100,'pilot_units':100,'color_nodes':count,
  'candidate_nodes':sum(bool(c) for c in colors.values()),'pilot_regions_with_candidate_color':sum(any(n.get('candidate_colors') for n in u['graph']['nodes']) for u in pilot),
  'file_sha256':{n:sha(ART/n) for n in ['pilot_units.jsonl.gz','train_units.jsonl.gz','candidate_records.jsonl.gz','membership.json']},
  'TEST_accessed':False,'evaluation_reference_text_read':False,'human_precision_gate_passed':False}
 save(ART/'complete.json',result);print(line(result),flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:
  if ART.exists():save(ART/'failure.json',{'utc':now(),'reason':str(e),'TEST_accessed':False})
  traceback.print_exc();raise
