"""Compact read-only active pilot status; never polls terminal SG/color phases."""
from pathlib import Path
import json
root=Path.home()/'rsg-former';art=root/'artifacts/vg_caption_pilot100_v1_20261026';result={}
for name in ['status.json','complete.json','failure.json']:
 if (art/name).exists():result[name]=json.loads((art/name).read_text())
for stage in ['prepare','generation','metrics']:
 entry={}
 for name in ['progress.json','complete.json','failure.json','prompt_contract.json']:
  p=art/stage/name
  if p.exists():
   d=json.loads(p.read_text());entry[name]={k:v for k,v in d.items() if k not in ['file_sha256','source_sha256','metrics_file_sha256']}
 if entry:result[stage]=entry
state=result.get('status.json',{})
for key in ['controller_pid','child_pid']:
 if state.get(key):result[key+'_proc_exists']=(Path('/proc')/str(state[key])).exists()
print(json.dumps(result))
