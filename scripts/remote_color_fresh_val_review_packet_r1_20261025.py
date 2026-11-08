"""Blind held-out review packet: no predicted colors in the annotation sheet."""
from pathlib import Path
from datetime import datetime,timezone
import collections,csv,gzip,hashlib,json,zipfile
root=Path.home()/'rsg-former';art=root/'artifacts/color_fresh_val_execution_r1_20261025';out=art/'blind_review_packet_v1'
assert (art/'complete.json').exists() and not out.exists();out.mkdir()
complete=json.loads((art/'complete.json').read_text());raw=art/'candidate_records.jsonl.gz'
assert hashlib.sha256(raw.read_bytes()).hexdigest()==complete['candidate_records_sha256']
with gzip.open(raw,'rt') as f:records=list(map(json.loads,f))
def colors(row):return row['methods']['SAM2_ColorNames'].get('operating_points',{}).get('conservative',{}).get('colors',[])
def binding(row):return row.get('native_object_binding') and row['native_object_binding']['profile_pass']['strict']
candidates=[r for r in records if r['partition']=='color_check_v1' and colors(r) and binding(r)]
ordered=sorted(candidates,key=lambda r:hashlib.sha256(('COLOR_CHECK_REVIEW_V1_20261025|'+r['key']).encode()).hexdigest())
selected=[];seen=set()
for r in ordered:
 if r['image_id'] in seen:continue
 seen.add(r['image_id']);selected.append(r)
 if len(selected)==200:break
sheet=[]
for r in selected:
 sheet.append({'key':r['key'],'image_id':r['image_id'],'target_class':r['class'],'image':'thumbnails/'+r['thumbnail'],
  'panels':'original | GrabCut mask | SAM2 mask','target_correct_yes_no_uncertain':'','mask_correct_yes_no_uncertain':'',
  'visible_main_color_or_unknown':'','visible_secondary_color_or_none':'','should_abstain_yes_no':'','reviewer':'','notes':''})
fields=list(sheet[0]) if sheet else ['key','image_id','target_class','image','target_correct_yes_no_uncertain','mask_correct_yes_no_uncertain','visible_main_color_or_unknown','visible_secondary_color_or_none','should_abstain_yes_no','reviewer','notes']
with (out/'blind_annotation.csv').open('x',newline='',encoding='utf-8-sig') as f:
 writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(sheet)
(out/'predictions_for_reveal_AFTER_annotation.json').write_text(json.dumps({r['key']:colors(r) for r in selected},indent=2))
(out/'README.txt').write_text('Independent color-check review. Annotate original target and mask before opening predictions_for_reveal_AFTER_annotation.json.\nPalette: black blue brown gray green orange pink purple red white yellow; use unknown/uncertain when evidence is insufficient.\nOne selected instance per image; every empty annotation remains pending. No agent inspection counts as a human label.\nA sample of at least100 accepted instances and original precision gate are required. This packet never authorizes TEST by itself.\n')
summary={'status':'FRESH_VAL_COLOR_CANDIDATE_CACHE_AND_BLIND_REVIEW_PACKET_COMPLETE_HUMAN_GATE_PENDING','utc':datetime.now(timezone.utc).isoformat(),
 'expected_nodes':len(records),'partition_nodes':dict(collections.Counter(r['partition'] for r in records)),
 'check_conservative_color_and_strict_binding_candidates':len(candidates),'review_instances_distinct_parent_images':len(selected),
 'minimum_review_instances_required':100,'enough_candidate_instances_to_start_required_review':len(selected)>=100,
 'human_annotations_completed':0,'production_accepted_colors':0,'TEST_accessed':False,'caption_generation_launched':False,
 'class_review_support':dict(collections.Counter(r['class'] for r in selected)),
 'sampling':'predeclared fixed hash over locked candidate set, one instance per parent, maximum200; not stratified by GT/reference',
 'perceptual_identity_audit_complete':False,'masked_pixel_inputs':'native crops; no gray/white fill enters pixel statistics',
 'selected_policy':'SAM2 small + original ColorNames conservative + native CLIP strict compatibility; passed means candidate, never certified correctness',
 'source_sha256':hashlib.sha256((root/'scripts/remote_color_fresh_val_review_packet_r1_20261025.py').read_bytes()).hexdigest(),
 'candidate_records_sha256':complete['candidate_records_sha256']}
(out/'summary.json').write_text(json.dumps(summary,indent=2))
with zipfile.ZipFile(out/'blind_review_packet.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
 for p in out.iterdir():
  if p.is_file() and p.suffix!='.zip':z.write(p,p.name)
 for r in selected:z.write(art/'thumbnails'/r['thumbnail'],'thumbnails/'+r['thumbnail'])
(root/'reports/VG_COLOR_FRESH_VAL_REVIEW_PACKET_R1_20261025.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary))
