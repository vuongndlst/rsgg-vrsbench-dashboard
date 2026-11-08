"""Immutable geometry-only TRAIN/check preparation, before any new generation."""
from pathlib import Path
import collections,gzip,json,os,signal
ROOT=Path.home()/'rsg-former';PIPE=ROOT/'artifacts/vg_focus_repair_v1_20261028';ART=PIPE/'prepare';BASE=ROOT/'artifacts/vg_caption_execution_v1_20261025'
os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:4]));signal.alarm(600)
from rsgformer.vg_color_execution_common_v1 import sha,save,line,now
from rsgformer.vg_caption_graph_v1 import digest,hash_order,focus_graph
from rsgformer.vg_region_focus_repair_v1 import select,synthetic_contract
assert not ART.exists();ART.mkdir()
lock=json.loads((PIPE/'preregistration.json').read_text())
for p,h in lock['source_sha256'].items():assert sha(ROOT/p)==h
contract=synthetic_contract();save(ART/'synthetic_contract.json',contract)
base=json.loads((BASE/'base_cache_complete.json').read_text())
for n in ['region_units_no_color.jsonl.gz','membership.json','train_bank_base_sg.jsonl.gz','val_base_sg.jsonl.gz']:assert sha(BASE/n)==base['file_sha256'][n]
membership=json.loads((BASE/'membership.json').read_text());old=json.loads((ROOT/'artifacts/vg_caption_pilot100_v1_20261026/prepare/membership.json').read_text())
assert len(membership['train_bank'])==200 and len(membership['caption_check'])==64
assert not set(membership['caption_check'])&set(old['pilot_VAL_images'])
assert not (BASE/'caption_check_v1').exists() and not (BASE/'caption_check_metrics_v1').exists(),'old_check_already_observed_requires_audit'
with gzip.open(BASE/'region_units_no_color.jsonl.gz','rt') as f:all_units=list(map(json.loads,f))
sets={'train':[u for u in all_units if u['partition']=='train_bank'],'check':[u for u in all_units if u['partition']=='caption_check']}
assert len(sets['train'])==800 and len(sets['check'])==256
assert {u['parent_image_id'] for u in sets['train']}==set(membership['train_bank'])
assert {u['parent_image_id'] for u in sets['check']}==set(membership['caption_check'])
want=set(membership['train_bank']+membership['caption_check']);graphs={}
for filename in ['train_bank_base_sg.jsonl.gz','val_base_sg.jsonl.gz']:
 with gzip.open(BASE/filename,'rt') as f:
  for row in map(json.loads,f):
   if row['image_id'] in want:assert digest(row['graph'])==row['base_graph_sha256'];graphs[row['image_id']]=row
assert set(graphs)==want;reproduced=0
for split,units in sets.items():
 with gzip.open(ART/(split+'_units.jsonl.gz'),'xt') as out:
  for u in sorted(units,key=lambda u:u['unit_id']):
   g=graphs[u['parent_image_id']]['graph'];assert digest(g)==u['base_graph_sha256'];assert select(g,u['crop_xyxy'],'L0')==u['graph'];reproduced+=1
   geometry={k:u[k] for k in ['unit_id','parent_image_id','region_id','image_size','crop_xyxy','scope','split','partition','base_graph_sha256']};out.write(line(geometry)+'\n')
 with gzip.open(ART/(split+'_graphs.jsonl.gz'),'xt') as out:
  for i in sorted({u['parent_image_id'] for u in units}):out.write(line(graphs[i])+'\n')
save(ART/'fixed_check_qualitative_sample_ids.json',sorted([u['unit_id'] for u in sets['check']],key=lambda k:hash_order('VG_FOCUS_REPAIR_CHECK_QUAL20_V1',k))[:20])
result={'status':'FOCUS_REPAIR_GEOMETRY_PREPARATION_COMPLETE','utc':now(),'TRAIN_units':800,'check_units':256,'legacy_views_reproduced':reproduced,'check_parent_images':64,
 'source_base_completion_sha256':sha(BASE/'base_cache_complete.json'),'file_sha256':{p.name:sha(p) for p in ART.iterdir() if p.is_file()},'GT_reference_text_read':False,'TEST_accessed':False}
save(ART/'complete.json',result);print(line(result))
