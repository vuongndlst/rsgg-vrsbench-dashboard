"""Exclusive bounded CPU preparation of fresh-dev geometry and separate references.
No detector, model, scoring, heldout-check export or TEST images/references.
"""
from pathlib import Path
from datetime import datetime,timezone
import gc,gzip,hashlib,json,math,os,resource,signal,shutil,sqlite3,time,traceback,zipfile
ROOT=Path.home()/'rsg-former';PIPE=ROOT/'artifacts/vg_grounding_dev_prepare_v1_20261102';BASE=ROOT/'artifacts/vg_caption_execution_v1_20261025'
CONFIG=ROOT/'configs/vg_grounding_controls_v1_20261102.json';MANIFEST=ROOT/'configs/vg_grounding_controls_source_manifest_v1_20261102.json'
START=time.time();os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});signal.alarm(180)
resource.setrlimit(resource.RLIMIT_AS,(768*1024**2,768*1024**2))
from rsgformer.vg_grounding_controls_v1_20261102 import geometry_only_units,synthetic_contract
from rsgformer.vg_region_focus_repair_v1 import select
from rsgformer.vg_caption_graph_v1 import digest

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return h.hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def save(name,value):
 with (PIPE/name).open('x',encoding='utf8') as f:json.dump(value,f,indent=2)
def guard():
 assert time.time()<1794670321.2414355 and time.time()-START<180,'deadline_or_wall_cap'
 assert shutil.disk_usage(ROOT).free>=16*1024**3,'disk_reserve'
 assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<768*1024**2,'RSS_cap'
 assert not (PIPE/'STOP').exists(),'owned_stop'
def items(path):
 decoder=json.JSONDecoder();buf=''
 with path.open(encoding='utf8') as f:
  assert f.read(1)=='['
  while True:
   guard();buf=buf.lstrip()
   if buf.startswith(','):buf=buf[1:].lstrip()
   if buf.startswith(']'):return
   try:item,end=decoder.raw_decode(buf)
   except json.JSONDecodeError:
    part=f.read(1024**2);assert part,'truncated_annotation_source';buf+=part;continue
   buf=buf[end:];yield item

def main():
 assert not PIPE.exists(),'exclusive_preparation_already_exists_no_retry'
 cfg=json.loads(CONFIG.read_text());manifest=json.loads(MANIFEST.read_text())
 for rel,h in manifest['sha256'].items():assert sha(ROOT/rel)==h,rel
 assert synthetic_contract()['status']=='SYNTHETIC_GROUNDING_CONTRACT_PASS'
 free=shutil.disk_usage(ROOT).free;seen=set();aggregate=0
 for p in (ROOT/'artifacts').rglob('*'):
  if p.is_symlink() or not p.is_file():continue
  st=p.stat();key=(st.st_dev,st.st_ino)
  if key not in seen:aggregate+=st.st_size;seen.add(key)
 assert free>=17*1024**3 and aggregate+1024**3<=40*1024**3,'storage_admission'
 ram=int(next(s for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')).split()[1])*1024
 assert ram>=1024**3,'CPU_RAM_admission'
 assert sha(BASE/'base_cache_complete.json')==cfg['expected_base_cache_completion_sha256'],'base_completion_changed'
 base_complete=json.loads((BASE/'base_cache_complete.json').read_text());valpath=BASE/'val_base_sg.jsonl.gz'
 assert sha(valpath)==base_complete['file_sha256'][valpath.name],'saved_VAL_graph_cache_changed'
 expected=set(cfg['fresh_dev_ids']);check=set(cfg['fresh_check_ids_not_to_load']);assert len(expected)==len(check)==64 and not expected&check
 dbpath=ROOT/'artifacts/caption_readiness/vg_pixel_fingerprints_v1.sqlite';dbhash=sha(dbpath)
 assert dbhash=='983789d6425c628ac22849090f6160f8b3e55910da4f1aecad8a5b61a288e6f9','pixel_index_changed'
 db=sqlite3.connect(dbpath.as_uri()+'?mode=ro',uri=True);db.execute('PRAGMA query_only=ON')
 for i in expected:
  row=db.execute('SELECT jpeg_sha,rgb_sha,width,height FROM pixels WHERE vg_id=?',(i,)).fetchone();p=cfg['fingerprints'][str(i)]
  assert row==(p['jpeg_sha256'],p['rgb_sha256'],p['width'],p['height'])
 db.close()
 regionpath=ROOT/'data/vg_caption/region_descriptions.json';reference_source_hash=sha(regionpath)
 guard();PIPE.mkdir()
 lock={'phase':cfg['preparation_phase'],'utc':now(),'manifest_sha256':sha(MANIFEST),'source_config_hashes':manifest['sha256'],'base_completion_sha256':sha(BASE/'base_cache_complete.json'),
       'VAL_saved_graph_file_sha256':sha(valpath),'reference_annotation_source_sha256':reference_source_hash,'fresh_dev_parent_ids':sorted(expected),'reserved_check_parent_ids_not_to_export':sorted(check),'resource_admission':{'disk_free_bytes':free,'aggregate_artifacts_bytes':aggregate,'RAM_available_bytes':ram},
       'new_model_or_scoring':False,'check_or_TEST_generation_allowed':False,'region_selection_uses_reference_text':False,'raw_annotation_scan_disclosure':'streaming JSON parser transiently decodes raw annotation records; only locked dev references may be extracted on second pass; no heldout reference is exported or used'}
 save('preregistration.json',lock)
 graphs={};count=0
 with gzip.open(valpath,'rt') as f:
  for r in map(json.loads,f):
   count+=1
   if r['image_id'] in expected:
    assert digest(r['graph'])==r['base_graph_sha256'];graphs[r['image_id']]=r
 assert count==5000 and set(graphs)==expected
 inputs=[];rejected=[];found=set()
 for item in items(regionpath):
  i=item['id']
  if i not in expected:continue
  assert i not in found;found.add(i);g=graphs[i]
  units,bad=geometry_only_units(i,[g['width'],g['height']],item['regions']);rejected.extend(bad)
  ids=sorted(units,key=lambda u:hashlib.sha256(('VG_GROUNDING_REGION_V1_20261102|'+u).encode()).hexdigest())[:4]
  assert len(ids)==4,'insufficient_valid_geometry_keep_parent_no_resample'
  for uid in ids:
   q=units[uid];q['base_graph_sha256']=g['base_graph_sha256'];q['graph']=select(g['graph'],q['crop_xyxy'],'L2');inputs.append(q)
 assert found==expected and len(inputs)==256
 assert all(u['parent_image_id'] not in check for u in inputs)
 with gzip.open(PIPE/'dev_units_unbounded_L2.jsonl.gz','xt',encoding='utf8') as f:
  for u in inputs:f.write(json.dumps(u)+'\n')
 with gzip.open(PIPE/'dev_full_graphs.jsonl.gz','xt',encoding='utf8') as f:
  for i in sorted(graphs):f.write(json.dumps(graphs[i])+'\n')
 save('rejected_geometry.json',rejected)
 save('region_membership_lock.json',{'utc':now(),'unit_ids':[u['unit_id'] for u in inputs],'parents':sorted(found),'geometry_only':True,'dev_units_sha256':sha(PIPE/'dev_units_unbounded_L2.jsonl.gz'),'reference_extraction_started':False,'check_units_exported':False})
 # Geometry membership already committed above. Reference content never changes it.
 chosen={u['unit_id'] for u in inputs};refs={};missing=[]
 for item in items(regionpath):
  i=item['id']
  if i not in expected:continue
  for r in item['regions']:
   uid='vg:'+str(i)+':region:'+str(r['region_id'])
   if uid not in chosen:continue
   assert uid not in refs;phrase=r.get('phrase');refs[uid]=[phrase] if isinstance(phrase,str) and phrase.strip() else []
   if not refs[uid]:missing.append(uid)
 assert set(refs)==chosen
 save('dev_references_scoring_only.json',refs);save('missing_reference_ids_retained.json',missing)
 assert sha(regionpath)==reference_source_hash and sha(dbpath)==dbhash,'immutable_source_changed'
 guard();files=[p for p in PIPE.iterdir() if p.is_file()];assert sum(p.stat().st_size for p in files)<32*1024**2
 result={'status':'FRESH_DEV_GEOMETRY_AND_SEPARATE_REFERENCES_COMPLETE_NO_GENERATION','utc':now(),'phase':cfg['preparation_phase'],'dev_parent_images':64,'dev_units':256,'reserved_check_images_unopened':64,'missing_reference_units_retained':len(missing),
 'region_membership_locked_before_reference_extraction':True,'region_membership_sha256':sha(PIPE/'region_membership_lock.json'),'preregistration_sha256':sha(PIPE/'preregistration.json'),'source_manifest_sha256':sha(MANIFEST),'file_sha256':{p.name:sha(p) for p in files},
 'raw_annotation_scanned':True,'heldout_check_reference_text_exported_or_used':False,'reference_used_to_select_correct_nodes':False,'query_graphs_unbounded_until_shared_frozen_tokenizer_contract':True,'new_model_inference_or_scoring':False,'TEST_images_predictions_or_references_exported_or_used':False,'check_generated':False,'policy_adopted':False,
 'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'wall_seconds':time.time()-START,'disk_free_bytes_after':shutil.disk_usage(ROOT).free}
 save('completion_evidence.json',result)
 archive=PIPE/'grounding_dev_preparation_results.zip'
 with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
  for p in [*files,PIPE/'completion_evidence.json']:z.write(p,p.name)
 assert archive.stat().st_size<64*1024**2;guard()
 result.update(archive=archive.relative_to(ROOT).as_posix(),archive_sha256=sha(archive),archive_bytes=archive.stat().st_size)
 save('complete.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:
  if PIPE.exists() and not (PIPE/'complete.json').exists() and not (PIPE/'failure.json').exists():save('failure.json',{'utc':now(),'reason':str(e),'status':'FAILED_NO_AUTOMATIC_RETRY','preserve_partial_geometry_and_reference_exposure':True})
  traceback.print_exc();raise
