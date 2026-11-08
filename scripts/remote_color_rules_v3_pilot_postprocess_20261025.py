"""Immutable engineering comparison/review packet, never substitutes for human GT."""
from pathlib import Path
from datetime import datetime,timezone
import collections,csv,gzip,hashlib,json,zipfile
root=Path.home()/'rsg-former';base=root/'artifacts/vg_caption_execution_v1_20261025';art=root/'artifacts/color_rules_v3_train_pilot_r1_20261025';out=art/'engineering_comparison'
assert (art/'complete.json').exists() and not out.exists();out.mkdir()
with gzip.open(art/'candidate_records.jsonl.gz','rt') as f:new=list(map(json.loads,f))
with gzip.open(base/'color_candidates_v2/color_candidates.jsonl.gz','rt') as f:old={r['key']:r for r in map(json.loads,f) if r['partition']=='train_bank'}
with gzip.open(base/'color_gt_matches_v1/color_gt_diagnostics_separate.jsonl.gz','rt') as f:gt={r['key']:r for r in map(json.loads,f)}
assert len(new)==len(old)==200 and {r['key'] for r in new}==set(old)
stats={};reviews=[]
for method in ['GrabCut_ColorNames','SAM2_ColorNames','SAM2_ColorNames_CLIP_contradiction_flag']:
 for op in ['balanced','conservative']:
  rows=[];contradictions=0
  for r in new:
   result=r['methods']['SAM2_ColorNames' if method.startswith('SAM2') else 'GrabCut_ColorNames']
   colors=result.get('operating_points',{}).get(op,{}).get('colors',[])
   clip=r.get('CLIP_fill127_fill255');flag=bool(colors and clip and all(s['top_color'] not in colors and s['margin']>=.03 for s in clip))
   contradictions+=flag
   # Third arm flags contradictions; it does not silently change colors or assert CLIP truth.
   if colors:rows.append((r,colors))
  labeled=[(r,c) for r,c in rows if gt[r['key']]['color_labels']]
  stats[method+':'+op]={'expected_nodes':200,'candidate_nodes':len(rows),'candidate_coverage':len(rows)/200,
   'candidate_with_explicit_positive_GT':len(labeled),'positive_GT_subset_agreement':sum(set(c)<=set(gt[r['key']]['color_labels']) for r,c in labeled)/len(labeled) if labeled else None,
   'agreement_is_NOT_visual_precision':True,'same_target_human_reviewed':0,'CLIP_both_fill_contradiction_flags':contradictions if method.startswith('SAM2') else None}
for r in new:
 sam=r['methods']['SAM2_ColorNames'];grab=r['methods']['GrabCut_ColorNames']
 reviews.append({'key':r['key'],'image_id':r['image_id'],'split':'TRAIN_ENGINEERING_NOT_HELDOUT','class':r['class'],
  'surface_group':r['surface_group'],'thumbnail':r.get('thumbnail',''),'panel_order':'original | GrabCut+LUT mask | SAM2+LUT mask',
  'V2_E4_conservative_colors':','.join(old[r['key']].get('candidates',{}).get('2',{}).get('E4_grabcut_pixels_clip_agreement',[])),
  'V3_GrabCut_conservative_colors':','.join(grab.get('operating_points',{}).get('conservative',{}).get('colors',[])),
  'V3_SAM2_conservative_colors':','.join(sam.get('operating_points',{}).get('conservative',{}).get('colors',[])),
  'SAM2_reason':sam['reason'],'correct_target_human':'','mask_correct_human':'','main_color_correct_human':'','secondary_color_correct_human':'','should_abstain_human':'','reviewer':'','notes':''})
with (out/'review_sheet_TRAIN_only.csv').open('x',newline='',encoding='utf-8-sig') as f:
 writer=csv.DictWriter(f,fieldnames=list(reviews[0]));writer.writeheader();writer.writerows(reviews)
report={'status':'TRAIN_COLOR_RULE_ENGINEERING_COMPARISON_COMPLETE_NO_VISUAL_ACCURACY_CLAIM','utc':datetime.now(timezone.utc).isoformat(),
 'nodes':200,'statistics':stats,'GT_only_in_diagnostic_report':True,'TEST_accessed':False,'human_reviews_completed':0,
 'CLIP_flag_definition':'both neutral fills top_color absent from pixel candidate set and each margin>=.03; diagnostic flag, not calibrated truth',
 'rule_version_gate_passed':False,'sampling':'same frozen200TRAIN nodes; every missing/unknown retained in expected denominator',
 'preregistration_sha256':hashlib.sha256((art/'preregistration.json').read_bytes()).hexdigest(),
 'candidate_records_sha256':hashlib.sha256((art/'candidate_records.jsonl.gz').read_bytes()).hexdigest()}
(out/'summary.json').write_text(json.dumps(report,indent=2));(root/'reports/VG_COLOR_RULES_V3_ENGINEERING_COMPARISON_20261025.json').write_text(json.dumps(report,indent=2))
with zipfile.ZipFile(out/'review_packet_TRAIN_only.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
 z.write(out/'review_sheet_TRAIN_only.csv','review_sheet_TRAIN_only.csv');z.write(out/'summary.json','summary.json')
 for r in reviews:
  if r['thumbnail']:z.write(art/'thumbnails'/r['thumbnail'],'thumbnails/'+r['thumbnail'])
print(json.dumps(report))
