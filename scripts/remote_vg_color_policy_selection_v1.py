"""Finite color-policy selection on dev; check is reported once, not tuned."""
from pathlib import Path
from datetime import datetime,timezone
import gzip,json,hashlib
from rsgformer.vg_caption_graph_v1 import hash_order
ROOT=Path.home()/'rsg-former';BASE=ROOT/'artifacts/vg_caption_execution_v1_20261025';ART=BASE/'color_candidates_v2'
assert (ART/'complete.json').exists() and not (BASE/'color_policy_selection_v1.json').exists()
with gzip.open(ART/'color_candidates.jsonl.gz','rt') as f:rows=[json.loads(x) for x in f]
with gzip.open(BASE/'color_gt_matches_v1/color_gt_diagnostics_separate.jsonl.gz','rt') as f:gt={r['key']:r for r in map(json.loads,f)}
methods=['E0_bbox_pixels','E1_grabcut_pixels','E2_clip_bbox','E3_grabcut_clip','E4_grabcut_pixels_clip_agreement'];stats={}
for partition in ['color_dev','color_check']:
 local=[r for r in rows if r['partition']==partition];pstats={}
 for method in methods:
  pstats[method]={}
  for op in ['0','1','2']:
   accepted=[r for r in local if r.get('candidates',{}).get(op,{}).get(method)]
   labeled=[r for r in accepted if gt[r['key']]['color_labels']]
   exact=sum(set(r['candidates'][op][method])<=set(gt[r['key']]['color_labels']) for r in labeled)
   pstats[method][op]={'target_nodes':len(local),'threshold_accepted':len(accepted),'coverage':len(accepted)/max(1,len(local)),
    'accepted_with_explicit_GT_color':len(labeled),'weak_annotated_label_agreement':exact/max(1,len(labeled)),
    'agreement_is_not_visual_precision':True,'unlabeled_not_false_positive':True}
 stats[partition]=pstats
selection={}
for method in methods:
 qualified=[(op,s) for op,s in stats['color_dev'][method].items() if s['accepted_with_explicit_GT_color']>=20 and s['weak_annotated_label_agreement']>=.9]
 op=max(qualified,key=lambda pair:(pair[1]['threshold_accepted'],-int(pair[0])))[0] if qualified else '2'
 selection[method]={'operating_point':op,'reason':'maximum coverage among predeclared weak-label dev gate' if qualified else 'no qualifying dev point; prespecified conservative point2 for VAL exploratory diagnostics only',
  'human_visual_precision_gate_passed':False,'test_color_allowed':False}
chosen=selection['E4_grabcut_pixels_clip_agreement']['operating_point']
check_accepted=[r for r in rows if r['partition']=='color_check' and r.get('candidates',{}).get(chosen,{}).get('E4_grabcut_pixels_clip_agreement')]
review=sorted(check_accepted,key=lambda r:hash_order('VG_COLOR_HELD_OUT_REVIEW_V1',r['key']))[:200]
# All pending review remains pending; an agent-generated contact sheet is not human annotation.
result={'status':'DEV_POLICY_SELECTION_COMPLETE_COLOR_TEST_GATE_PENDING','utc':datetime.now(timezone.utc).isoformat(),'statistics':stats,'selection':selection,
 'caption_color_policy_for_VAL_exploratory_only':{'method':'E4_grabcut_pixels_clip_agreement','operating_point':chosen},
 'human_visual_review_keys':[r['key'] for r in review],'human_reviews_completed':0,'color_accepted_for_TEST':False,'test_accessed':False}
(BASE/'color_policy_selection_v1.json').write_text(json.dumps(result,indent=2));(ROOT/'reports/VG_COLOR_POLICY_SELECTION_20261025.json').write_text(json.dumps(result,indent=2))
print(json.dumps({'status':result['status'],'selected_VAL_candidate_op':chosen,'color_check_accepted_nodes':len(check_accepted),'human_visual_gate':'PENDING_NOT_FABRICATED','dev_statistics':stats['color_dev'],'test_color_allowed':False}))
