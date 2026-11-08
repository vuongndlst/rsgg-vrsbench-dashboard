"""Read-only compact status for the active automated color phase, no terminal SG poll."""
from pathlib import Path
import json,os
ROOT=Path.home()/'rsg-former';ART=ROOT/'artifacts/vg_color_auto_full_pipeline_v1_20261025'
result={}
for name in ['status.json','failure.json','complete.json']:
 p=ART/name
 if p.exists():result[name]=json.loads(p.read_text())
for name in ['vg_color_fast_val_replay_v1_20261025','vg_color_full_test_auto_v1_20261025','vg_color_auto_test_diagnostic_v1_20261025']:
 folder=ROOT/'artifacts'/name;item={}
 for f in ['progress.json','complete.json','failure.json']:
  if (folder/f).exists():
   data=json.loads((folder/f).read_text());item[f]={k:v for k,v in data.items() if k not in ['by_class','differences','source_sha256']}
 if item:result[name]=item
state=result.get('status.json',{})
for key in ['controller_pid','child_pid']:
 pid=state.get(key)
 if pid:result[key+'_proc_exists']=(Path('/proc')/str(pid)).exists()
print(json.dumps(result))
