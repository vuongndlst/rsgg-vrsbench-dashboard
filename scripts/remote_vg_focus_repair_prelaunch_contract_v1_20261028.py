"""CPU-only prelaunch contracts while Qwen resource admission is unavailable."""
from pathlib import Path
import collections,gzip,json,os,resource,shutil,signal,zipfile
ROOT=Path.home()/'rsg-former';BASE=ROOT/'artifacts/vg_caption_execution_v1_20261025';OUT=ROOT/'artifacts/vg_focus_repair_prelaunch_contract_v1_20261028';PIPE=ROOT/'artifacts/vg_focus_repair_v1_20261028'
os.sched_setaffinity(0,{sorted(os.sched_getaffinity(0))[0]});signal.alarm(120)
from rsgformer.vg_color_execution_common_v1 import sha,save,line,now
from rsgformer.vg_caption_graph_v1 import digest
from rsgformer.vg_region_focus_repair_v1 import select,synthetic_contract,POLICIES
assert not OUT.exists() and not PIPE.exists() and not (ROOT/'reports/VG_FOCUS_REPAIR_LAUNCH_20261028.json').exists(),'main_experiment_must_remain_unlaunched'
assert shutil.disk_usage(ROOT).free>=16*1024**3+1024**2
manifest=json.loads((ROOT/'configs/vg_focus_repair_source_manifest_v1_20261028.json').read_text())
for p,h in manifest['source_sha256'].items():assert sha(ROOT/p)==h
contract=synthetic_contract();base=json.loads((BASE/'base_cache_complete.json').read_text())
for n in ['region_units_no_color.jsonl.gz','train_bank_base_sg.jsonl.gz','val_base_sg.jsonl.gz']:assert sha(BASE/n)==base['file_sha256'][n]
with gzip.open(BASE/'region_units_no_color.jsonl.gz','rt') as f:units=[u for u in map(json.loads,f) if u['partition'] in ['train_bank','caption_check']]
assert collections.Counter(u['partition'] for u in units)=={'train_bank':800,'caption_check':256}
wanted={u['parent_image_id'] for u in units};graphs={}
for n in ['train_bank_base_sg.jsonl.gz','val_base_sg.jsonl.gz']:
 with gzip.open(BASE/n,'rt') as f:
  for row in map(json.loads,f):
   if row['image_id'] in wanted:assert digest(row['graph'])==row['base_graph_sha256'];graphs[row['image_id']]=row
count=0
for u in units:
 g=graphs[u['parent_image_id']]['graph'];assert digest(g)==u['base_graph_sha256'];assert select(g,u['crop_xyxy'],'L0')==u['graph']
 for policy in POLICIES:
  view=select(g,u['crop_xyxy'],policy);ids={n['id'] for n in view['nodes']};assert len(ids)<=24 and len(view['edges'])<=32
  assert all(n in g['nodes'] for n in view['nodes']) and all(e in g['edges'] for e in view['edges'])
  assert all(e['subject'] in ids and e['object'] in ids for e in view['edges'])
  if policy=='L1' and u['graph']['nodes']:assert view==u['graph']
  count+=1
assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=768*1024**2
OUT.mkdir();result={'status':'PRELAUNCH_SOURCE_NATIVE1056_AND3168_STRUCTURE_CONTRACT_PASS','utc':now(),'source_manifest_sha256':sha(ROOT/'configs/vg_focus_repair_source_manifest_v1_20261028.json'),
 'source_sha256':manifest['source_sha256'],'TRAIN_regions':800,'check_geometry_regions':256,'exact_L0_reproductions':1056,'structural_cases':count,'synthetic':contract,
 'check_reference_text_read':False,'reference_or_GT_attributes_or_images_read':False,'TEST_accessed':False,'model_or_metrics_run':False,'new_policy_selected_or_adopted':False,
 'experiment_namespace_exists':PIPE.exists(),'remote_launch_report_exists':(ROOT/'reports/VG_FOCUS_REPAIR_LAUNCH_20261028.json').exists(),'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
save(OUT/'complete.json',result)
archive=OUT/'contract_evidence.zip'
with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:z.write(OUT/'complete.json','complete.json')
result.update(archive=str(archive.relative_to(ROOT)),archive_sha256=sha(archive));print(line(result))
