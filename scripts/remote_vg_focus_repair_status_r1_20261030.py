"""Compact active-only focus experiment status, no terminal historical polling."""
from pathlib import Path
import json
root=Path.home()/'rsg-former';art=root/'artifacts/vg_focus_repair_r1_20261030';result={}
for name in ['status.json','complete.json','failure.json','selection.json']:
 if (art/name).exists():result[name]=json.loads((art/name).read_text())
for stage in ['prepare','train_generation','train_metrics','check_generation','check_metrics']:
 item={}
 for name in ['progress.json','complete.json','failure.json','prompt_contract.json']:
  p=art/stage/name
  if p.exists():item[name]={k:v for k,v in json.loads(p.read_text()).items() if k not in ['file_sha256','metric_sha256']}
 if item:result[stage]=item
s=result.get('status.json',{})
for k in ['controller_pid','child_pid']:
 if s.get(k):result[k+'_proc_exists']=(Path('/proc')/str(s[k])).exists()
print(json.dumps(result))
