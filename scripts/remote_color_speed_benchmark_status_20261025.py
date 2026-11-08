"""Read-only speed benchmark status and bounded errors."""
from pathlib import Path
from datetime import datetime,timezone
import json
root=Path.home()/'rsg-former';art=root/'artifacts/vg_color_speed_benchmark_v1_20261025';out={'utc':datetime.now(timezone.utc).isoformat()}
for name in ['progress.json','failure.json','complete.json']:
 p=art/name
 if p.exists():out[name]=json.loads(p.read_text())
p=root/'reports/VG_COLOR_SPEED_BENCHMARK_LAUNCH_20261025.json'
if p.exists():
 launch=json.loads(p.read_text());out['launch']=launch;out['pid_present']=(Path('/proc')/str(launch['pid'])).exists()
p=root/'reports/VG_COLOR_SPEED_BENCHMARK_20261025.stderr.log'
if p.exists():out['stderr_tail']=p.read_text()[-2200:]
print(json.dumps(out))
