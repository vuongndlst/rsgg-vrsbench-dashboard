"""Compact read-only active color validation status; no terminal SGDet polling."""
from pathlib import Path
from datetime import datetime,timezone
import json
root=Path.home()/'rsg-former';art=root/'artifacts/color_fresh_val_execution_r1_20261025';out={'utc':datetime.now(timezone.utc).isoformat()}
for name in ['progress.json','failure.json','complete.json']:
 p=art/name
 if p.exists():out[name]=json.loads(p.read_text())
for name in ['VG_COLOR_FRESH_VAL_CONTROLLER_R1_20261025.json','VG_COLOR_FRESH_VAL_REVIEW_PACKET_R1_20261025.json']:
 p=root/'reports'/name
 if p.exists():out[name]=json.loads(p.read_text())
controller=out.get('VG_COLOR_FRESH_VAL_CONTROLLER_R1_20261025.json',{})
if controller.get('child_pid'):out['worker_pid_present']=(Path('/proc')/str(controller['child_pid'])).exists()
for stem in ['remote_vg_color_fresh_val_execution_r1_20261025','remote_color_fresh_val_review_packet_r1_20261025']:
 p=root/('reports/'+stem+'.stderr.log')
 if p.exists():out[stem+'_stderr_tail']=p.read_text()[-1400:]
print(json.dumps(out))
