"""Compact read-only status of the new caption cache phase only."""
from pathlib import Path
import json
root=Path.home()/'rsg-former';art=root/'artifacts/vg_caption_execution_v1_20261025';out={}
for name in ['progress.json','cache_failure.json','base_cache_complete.json','color_cache_complete.json','caption_dev_result.json']:
 p=art/name
 if p.exists():
  d=json.loads(p.read_text());out[name]={k:v for k,v in d.items() if k not in ['file_sha256','source_hashes','units','per_unit']}
launch=root/'reports/VG_CAPTION_CACHE_LAUNCH_20261025.json'
if launch.exists():
 d=json.loads(launch.read_text());out['cache_pid']=d['pid'];out['cache_pid_present']=(Path('/proc')/str(d['pid'])).exists()
for name in ['progress.json','failure.json','complete.json']:
 p=art/'color_candidates_v2'/name
 if p.exists():out['color_'+name]=json.loads(p.read_text())
color_err=root/'reports/VG_COLOR_CACHE_V2_20261025.stderr.log'
if color_err.exists():out['color_stderr_tail']=color_err.read_text()[-1800:]
err=root/'reports/VG_CAPTION_CACHE_WORKER_20261025.stderr.log'
if err.exists():out['stderr_tail']=err.read_text()[-1800:]
print(json.dumps(out))
