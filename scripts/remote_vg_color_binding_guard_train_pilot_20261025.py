"""Frozen TRAIN target-binding diagnostic for V3 color candidates, no TEST."""
from pathlib import Path
from datetime import datetime,timezone
import csv,gzip,hashlib,io,json,os,resource,shutil,sqlite3,sys,time,zipfile
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',CUDA_CACHE_DISABLE='1')
ROOT=Path.home()/'rsg-former';BASE=ROOT/'artifacts/color_rules_v3_train_pilot_r1_20261025';ART=ROOT/'artifacts/color_object_binding_guard_train_v1_20261025'
assert (BASE/'complete.json').exists() and not ART.exists();assert shutil.disk_usage(ROOT).free>=17*1024**3
os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:4]));ART.mkdir()
cfg=json.loads((ROOT/'configs/vg_color_object_binding_guard_v1_20261025.json').read_text())
classes=json.loads((ROOT/'configs/vg_object_color_rules_v3_20261025.json').read_text())['class_rules']
with gzip.open(BASE/'candidate_records.jsonl.gz','rt') as f:rows=list(map(json.loads,f))
targets=[r for r in rows if r['methods']['SAM2_ColorNames'].get('operating_points',{}).get('conservative',{}).get('colors')]
assert len(rows)==200
lock={'phase':cfg['version'],'utc':datetime.now(timezone.utc).isoformat(),'TRAIN_only':True,'source_sha256':hashlib.sha256((ROOT/'scripts/remote_vg_color_binding_guard_train_pilot_20261025.py').read_bytes()).hexdigest(),
 'config_sha256':hashlib.sha256((ROOT/'configs/vg_color_object_binding_guard_v1_20261025.json').read_bytes()).hexdigest(),
 'parent_candidate_records_sha256':hashlib.sha256((BASE/'candidate_records.jsonl.gz').read_bytes()).hexdigest(),'target_candidates':len(targets),
 'accepted_production_attributes':False,'human_labels_used_for_binding':False,'TEST_access_allowed':False}
(ART/'preregistration.json').write_text(json.dumps(lock,indent=2))
import numpy as np,torch
from PIL import Image
from transformers import CLIPModel,CLIPProcessor
torch.set_num_threads(4);torch.set_num_interop_threads(1)
torch.cuda.set_per_process_memory_fraction(6*1024**3/torch.cuda.get_device_properties(0).total_memory)
assert torch.cuda.mem_get_info()[0]>=8*1024**3
folder=Path.home()/'shared-assets/models/clip-vit-base-patch32';model=CLIPModel.from_pretrained(folder,local_files_only=True).to('cuda').eval();processor=CLIPProcessor.from_pretrained(folder,local_files_only=True)
labels=sorted(classes)+cfg['distractors'];assert len(labels)==len(set(labels))
prompts=[t.format(object=cfg['aliases'].get(name,name)) for name in labels for t in cfg['templates']]
features=[]
with torch.inference_mode():
 for start in range(0,len(prompts),64):
  inputs=processor(text=prompts[start:start+64],padding=True,return_tensors='pt').to('cuda');f=model.get_text_features(**inputs)
  if not isinstance(f,torch.Tensor):f=f.pooler_output
  features.append(torch.nn.functional.normalize(f.float(),dim=-1))
 text=torch.nn.functional.normalize(torch.cat(features).reshape(len(labels),2,-1).mean(1),dim=-1)
wanted={r['image_id'] for r in targets};archives=[zipfile.ZipFile(Path.home()/'sbir/hienlth/vg'/n) for n in ['images.zip','images2.zip']];members={}
for z in archives:
 for m in z.infolist():
  if not m.filename.lower().endswith('.jpg'):continue
  try:i=int(Path(m.filename).stem)
  except ValueError:continue
  if i in wanted:assert i not in members;members[i]=(z,m)
assert set(members)==wanted
decisions=[]
for row in targets:
 assert time.time()<1794670321.2414355 and shutil.disk_usage(ROOT).free>=16*1024**3
 z,m=members[row['image_id']];blob=z.read(m);assert hashlib.sha256(blob).hexdigest()==row['jpeg_sha256']
 image=Image.open(io.BytesIO(blob)).convert('RGB');b=row['bbox_xyxy'];b=[max(0,int(b[0])),max(0,int(b[1])),min(image.width,int(np.ceil(b[2]))),min(image.height,int(np.ceil(b[3])))]
 original=image.crop(b);sheet=Image.open(BASE/'thumbnails'/row['thumbnail']).convert('RGB');masked=sheet.crop((320,0,480,160))
 with torch.inference_mode():
  f=model.get_image_features(**processor(images=[original,masked],return_tensors='pt').to('cuda'))
  if not isinstance(f,torch.Tensor):f=f.pooler_output
  scores=(torch.nn.functional.normalize(f.float(),dim=-1)@text.T).cpu().numpy()
 family={row['class']}
 for group in cfg['allowed_label_families']:
  if row['class'] in group:family.update(group)
 views=[]
 for vector in scores:
  order=np.argsort(-vector);position={labels[int(j)]:rank+1 for rank,j in enumerate(order)}
  best_family=max(family,key=lambda name:float(vector[labels.index(name)]));gap=float(vector[int(order[0])]-vector[labels.index(best_family)])
  views.append({'top5':[{'class':labels[int(j)],'cosine':float(vector[j])} for j in order[:5]],'best_target_family_member':best_family,'family_rank':position[best_family],'gap_to_best':gap})
 profiles={p['id']:all(v['family_rank']<=p['family_rank_max'] and v['gap_to_best']<=p['maximum_gap_to_best'] for v in views) for p in cfg['guard_profiles']}
 decisions.append({'key':row['key'],'class':row['class'],'colors':row['methods']['SAM2_ColorNames']['operating_points']['conservative']['colors'],'views':views,
  'binding_profile_pass':profiles,'attribute_status':'candidate_pending_human_validation' if profiles['strict'] else 'unknown_object_binding','production_accepted':False})
 assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=4*1024**3
for z in archives:z.close()
(ART/'decisions.json').write_text(json.dumps(decisions,indent=2))
result={'status':'TRAIN_OBJECT_BINDING_GUARD_DIAGNOSTIC_COMPLETE_NOT_VALIDATED','utc':datetime.now(timezone.utc).isoformat(),'initial_nodes':len(rows),
 'SAM2_conservative_color_candidates':len(targets),'strict_binding_remaining_candidates':sum(r['binding_profile_pass']['strict'] for r in decisions),
 'loose_binding_remaining_candidates':sum(r['binding_profile_pass']['diagnostic_loose'] for r in decisions),'human_reviews_completed':0,'production_accepted':False,
 'rejected_examples':[{'key':r['key'],'original_predicted_class':r['class'],'raw_clip_top':r['views'][0]['top5'][0]['class'],'mask_clip_top':r['views'][1]['top5'][0]['class']} for r in decisions if not r['binding_profile_pass']['strict']],
 'decisions_sha256':hashlib.sha256((ART/'decisions.json').read_bytes()).hexdigest(),'peak_allocator_bytes':torch.cuda.max_memory_allocated(),
 'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'TEST_accessed':False}
(ART/'complete.json').write_text(json.dumps(result,indent=2));(ROOT/'reports/VG_COLOR_OBJECT_BINDING_GUARD_TRAIN_20261025.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
