"""Freeze independent color dev/check membership before pixel inference; no TEST text."""
from pathlib import Path
from datetime import datetime,timezone
import collections,gzip,hashlib,json,sqlite3
ROOT=Path.home()/'rsg-former';BASE=ROOT/'artifacts/vg_caption_execution_v1_20261025';ART=ROOT/'artifacts/color_fresh_val_manifest_v1_20261025'
assert not ART.exists();ART.mkdir()
old=json.loads((BASE/'membership.json').read_text());old_ids={i for values in old.values() for i in values}
with sqlite3.connect((ROOT/'artifacts/caption_readiness/vg_pixel_fingerprints_v1.sqlite').as_uri()+'?mode=ro',uri=True) as db:
 pixels={i:(rgb,split) for i,rgb,split in db.execute('SELECT vg_id,rgb_sha,split FROM pixels')}
seen_pixels={pixels[i][0] for i in old_ids};records={}
with gzip.open(BASE/'val_base_sg.jsonl.gz','rt') as f:
 for row in map(json.loads,f):
  i=row['image_id'];assert pixels[i][1]=='val'
  if i not in old_ids and pixels[i][0] not in seen_pixels:records[i]=row
order=sorted(records,key=lambda i:hashlib.sha256(('VG_COLOR_FRESH_VAL_MEMBERSHIP_V1_20261025|'+str(i)).encode()).hexdigest())
chosen=[];unique_pixels=set()
for i in order:
 rgb=pixels[i][0]
 if rgb in unique_pixels:continue
 unique_pixels.add(rgb);chosen.append(i)
 if len(chosen)==1000:break
assert len(chosen)==1000
membership={'color_dev_v1':chosen[:200],'color_check_v1':chosen[200:]}
allowed=set('shirt jacket pant short coat hat cap helmet shoe bag bus truck car train cat dog horse cow plate toilet'.split())
targets=[];no_targets=[]
for partition,images in membership.items():
 for i in images:
  graph=records[i]['graph'];endpoints={e[k] for e in graph['edges'][:20] for k in ['subject','object']}
  nodes=[n for n in graph['nodes'] if n['label'] in allowed]
  nodes=sorted(nodes,key=lambda n:(n['id'] not in endpoints,-n['object_score'],n['id']))[:4]
  if not nodes:no_targets.append({'image_id':i,'partition':partition,'reason':'no_eligible_predicted_class'})
  for n in nodes:targets.append({'key':str(i)+':'+n['id'],'image_id':i,'partition':partition,'node':n,'RGB_identity_sha256':pixels[i][0],
   'top20_relation_endpoint':n['id'] in endpoints,'selection_has_reference_or_color_input':False})
def write(name,x):(ART/name).write_text(json.dumps(x,indent=2))
write('membership.json',membership);write('target_manifest.json',targets);write('images_without_target.json',no_targets)
report={'status':'FRESH_COLOR_VAL_DEV_CHECK_MEMBERSHIP_LOCKED_NO_INFERENCE','utc':datetime.now(timezone.utc).isoformat(),'images':1000,
 'dev_images':200,'check_images':800,'maximum_nodes_per_image':4,'targets':len(targets),'partition_target_counts':dict(collections.Counter(r['partition'] for r in targets)),
 'images_without_eligible_target_retained':len(no_targets),'disjoint_all_prior_engineering_parents_and_known_exact_pixels':True,
 'known_perceptual_identity_audit_complete':False,'class_scope':sorted(allowed),'TEST_attributes_or_reference_text_read':False,
 'selection_uses_predicted_inputs_only':True,'pixel_or_mask_inference_run':False,'color_policy_selected_from_this_check':False,
 'check_inference_requires_prior_locked_policy_and_resource_admission':True,'human_review_gate_pending':True,
 'source_sha256':hashlib.sha256((ROOT/'scripts/remote_prepare_color_fresh_val_manifest_v1_20261025.py').read_bytes()).hexdigest(),
 'manifest_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ART.iterdir() if p.is_file()}}
write('preregistration.json',report);(ROOT/'reports/VG_COLOR_FRESH_VAL_MANIFEST_20261025.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
