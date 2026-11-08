"""Read-only expanded historical-exposure identity reservation; no image, reference, caption or model input read."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,resource,signal,sqlite3,time
ROOT=Path.home()/'rsg-former'
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});signal.alarm(75)
resource.setrlimit(resource.RLIMIT_AS,(768*1024**2,768*1024**2))
start=time.time();assert start<1794670321.2414355,'absolute_deadline'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1024**2),b''):h.update(block)
 return h.hexdigest()
def order(i):return hashlib.sha256(('VG_GROUNDING_FRESH_VAL_V1_20261101|'+str(i)).encode()).hexdigest()
base=ROOT/'artifacts/vg_caption_execution_v1_20261025'
membership=json.loads((base/'membership.json').read_text())
old_val=set(sum([membership[k] for k in ['caption_dev','caption_check','color_dev','color_check']],[]));assert len(old_val)==528
train=set(membership['train_bank']);assert len(train)==200
fresh_path=ROOT/'artifacts/color_fresh_val_manifest_v1_20261025/membership.json'
assert sha(fresh_path)=='23235f2ef18fe99e6126118b607fd74a43c847344460a069d61dc317ba8d9b28','historical_color_membership_changed'
fresh=json.loads(fresh_path.read_text());historical_color=set()
for key,ids in fresh.items():
 if isinstance(ids,list):
  assert all(isinstance(i,int) for i in ids),'unexpected_color_membership_schema'
  historical_color.update(ids)
assert len(historical_color)==1000
old_val.update(historical_color)
path=ROOT/'artifacts/caption_readiness/vg_pixel_fingerprints_v1.sqlite'
before=sha(path);st=path.stat()
db=sqlite3.connect(path.as_uri()+'?mode=ro',uri=True,timeout=5);db.execute('PRAGMA query_only=ON')
rows=db.execute('SELECT vg_id,split,jpeg_sha,rgb_sha,width,height FROM pixels').fetchall();db.close()
assert len(rows)==89169
val={r[0]:r for r in rows if r[1]=='val'}
assert len(val)==5000 and old_val<=set(val) and train<=set(r[0] for r in rows)
blocked=[r for r in rows if r[1]!='val' or r[0] in old_val]
bjpeg={r[2] for r in blocked};brgb={r[3] for r in blocked}
candidates=[r for r in val.values() if r[0] not in old_val and r[2] not in bjpeg and r[3] not in brgb]
selected=[];seen_jpeg=set();seen_rgb=set()
for r in sorted(candidates,key=lambda r:order(r[0])):
 if r[2] in seen_jpeg or r[3] in seen_rgb:continue
 selected.append(r);seen_jpeg.add(r[2]);seen_rgb.add(r[3])
 if len(selected)==128:break
assert len(selected)==128 and sha(path)==before and path.stat().st_mtime_ns==st.st_mtime_ns
assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<768*1024**2
print(json.dumps({'status':'COMPLETE_READONLY_PROVISIONAL_FRESH_VAL_IDENTITY_RESERVATION','utc':datetime.now(timezone.utc).isoformat(),'phase':'vg_grounding_diagnostic_v1_20261101','membership_version':'v2_expanded_color_exposure_exclusion',
 'original_membership_sha256':sha(base/'membership.json'),'original_membership':membership,'pixel_database_sha256_before_after':before,'pixel_database_unchanged':True,'fingerprint_rows':len(rows),'canonical_VAL_IDs_sha256':hashlib.sha256(json.dumps(sorted(val),separators=(',',':')).encode()).hexdigest(),
 'historical_VAL_excluded':len(old_val),'historical_color1000_membership_sha256':sha(fresh_path),'historical_color_membership':fresh,'all_nonVAL_known_JPEG_RGB_identity_excluded':True,'eligible_VAL_images_before_group_unique':len(candidates),
 'fresh_dev':[{'image_id':r[0],'jpeg_sha256':r[2],'rgb_sha256':r[3],'width':r[4],'height':r[5]} for r in selected[:64]],
 'fresh_check':[{'image_id':r[0],'jpeg_sha256':r[2],'rgb_sha256':r[3],'width':r[4],'height':r[5]} for r in selected[64:]],
 'selection_read_reference_text':False,'images_models_graphs_or_captions_read':False,'remote_files_created_or_modified':False,'TEST_images_predictions_or_references_read':False,'protected_nonVAL_fingerprint_metadata_only_read':True,
 'perceptual_audit_complete':False,'caption_clean_certified':False,'membership_status':'PROVISIONAL_REFERENCE_BLIND_IDS_NO_REGION_CACHE_OR_HELDOUT_GATE_YET',
 'wall_seconds':time.time()-start,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}))
