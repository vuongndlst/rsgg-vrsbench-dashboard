"""Compact status of the bounded TRAIN color-rule pilot only."""
from pathlib import Path
from datetime import datetime,timezone
import json
root=Path.home()/'rsg-former';art=root/'artifacts/color_rules_v4_train_pilot_20261025';out={'utc':datetime.now(timezone.utc).isoformat()}
for name in ['progress.json','failure.json','complete.json']:
 p=art/name
 if p.exists():out[name]=json.loads(p.read_text())
launch=root/'reports/VG_COLOR_RULES_V4_TRAIN_PILOT_LAUNCH_20261025.json'
if launch.exists():
 record=json.loads(launch.read_text());out['launch']=record;out['pid_present']=(Path('/proc')/str(record['pid'])).exists()
err=root/'reports/VG_COLOR_RULES_V4_TRAIN_PILOT_20261025.stderr.log'
if err.exists():out['stderr_tail']=err.read_text()[-2300:]
print(json.dumps(out))
