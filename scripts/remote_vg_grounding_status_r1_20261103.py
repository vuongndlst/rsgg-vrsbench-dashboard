"""Single compact active-only status; never launches or reads heldout references."""
from pathlib import Path
import json
root=Path.home()/'rsg-former';art=root/'artifacts/vg_grounding_train_dev_r1_20261103';result={}
for name in ['status.json','complete.json','failure.json','selection_from_dev.json']:
 p=art/name
 if p.exists():result[name]=json.loads(p.read_text())
for stage in ['train_generation','train_metrics','dev_generation','dev_metrics']:
 data={}
 for name in ['progress.json','complete.json','failure.json']:
  p=art/stage/name
  if p.exists():data[name]={k:v for k,v in json.loads(p.read_text()).items() if k not in ['source_sha256','file_sha256','metric_sha256']}
 if data:result[stage]=data
s=result.get('status.json',{})
for k in ['controller_pid','child_pid']:
 if s.get(k):result[k+'_proc_exists']=(Path('/proc')/str(s[k])).exists()
print(json.dumps(result))
