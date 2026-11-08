"""Make compact deterministic TRAIN/VAL mask evidence for engineering inspection."""
from pathlib import Path
import gzip,json,zipfile,hashlib
root=Path.home()/'rsg-former';base=root/'artifacts/vg_caption_execution_v1_20261025';out=root/'artifacts/color_research_v3_20261025';out.mkdir(exist_ok=True)
with gzip.open(base/'color_candidates_v2/color_candidates.jsonl.gz','rt') as f:rows=list(map(json.loads,f))
with gzip.open(base/'color_gt_matches_v1/color_gt_diagnostics_separate.jsonl.gz','rt') as f:gt={r['key']:r for r in map(json.loads,f)}
selected=[];seen=set()
for category,predicate in [
 ('E4_disagrees',lambda r:bool(gt[r['key']]['color_labels']) and bool(r.get('candidates',{}).get('0',{}).get('E4_grabcut_pixels_clip_agreement')) and not set(r['candidates']['0']['E4_grabcut_pixels_clip_agreement'])<=set(gt[r['key']]['color_labels'])),
 ('E4_agrees',lambda r:bool(gt[r['key']]['color_labels']) and bool(r.get('candidates',{}).get('0',{}).get('E4_grabcut_pixels_clip_agreement')) and set(r['candidates']['0']['E4_grabcut_pixels_clip_agreement'])<=set(gt[r['key']]['color_labels'])),
 ('mask_failure',lambda r:r['status']=='mask_quality_failure')]:
 eligible=sorted([r for r in rows if r.get('thumbnail') and predicate(r)],key=lambda r:hashlib.sha256(('color_error_inspection_v3|'+r['key']).encode()).hexdigest())
 for r in eligible[:8]:
  if r['key'] in seen:continue
  seen.add(r['key']);selected.append({'category':category,**r,'GT_diagnostic_separate':gt[r['key']]})
(out/'v2_engineering_inspection.json').write_text(json.dumps(selected,indent=2))
with zipfile.ZipFile(out/'v2_engineering_inspection.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
 z.write(out/'v2_engineering_inspection.json','manifest.json')
 for r in selected:z.write(base/'color_candidates_v2/thumbnails'/r['thumbnail'],r['thumbnail'])
packages=[]
for p in [Path.home()/'etechs/thesis/.venv/lib/python3.10/site-packages/sam2',Path.home()/'etechs/thesis/.venv/lib/python3.12/site-packages/sam2',Path.home()/'etechs/thesis/vendor/sam2',Path.home()/'etechs/thesis/third_party/sam2']:
 if p.exists():packages.append(str(p))
print(json.dumps({'status':'ENGINEERING_TRAIN_VAL_PACKET_CREATED_NOT_HUMAN_REVIEW','items':len(selected),'archive_bytes':(out/'v2_engineering_inspection.zip').stat().st_size,'archive_sha256':hashlib.sha256((out/'v2_engineering_inspection.zip').read_bytes()).hexdigest(),'existing_sam2_packages':packages,'test_accessed':False}))
