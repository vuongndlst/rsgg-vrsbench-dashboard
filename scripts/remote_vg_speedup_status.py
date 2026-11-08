"""Compact read-only status of the once-only benchmark worker."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
root=Path.home()/'rsg-former';folder=root/'artifacts/caption_readiness/vg_speedup_controller_v1'
result={'utc':datetime.now(timezone.utc).isoformat(),'live_modified_by_status':False}
for name in ['controller','budget_guard']:
    p=folder/(name+'.pid')
    if p.exists():
        pid=int(p.read_text());cmd=Path('/proc')/str(pid)/'cmdline'
        token=b'remote_vg_speedup_controller.py' if name=='controller' else b'remote_vg_original_budget_guard.py'
        result[name]={'pid':pid,'active':cmd.exists() and token in cmd.read_bytes()}
status=folder/'status.json'
if status.exists():result['worker_status']=json.loads(status.read_text())
for report_name in ['VG_SPEEDUP_CONTROLLER.json','VG_SPEEDUP_BENCHMARK.json']:
    p=root/'reports'/report_name
    if not p.exists():continue
    raw=p.read_bytes();report=json.loads(raw)
    summary={k:report[k] for k in ['status','reason','selected_case','accepted','adoption_ready','training_adopted','resumed_pid','scientific_change_applied','original_wall_deadline_unix','next_action'] if k in report}
    summary['evidence_sha256']=hashlib.sha256(raw).hexdigest()
    result[report_name]=summary
print(json.dumps(result))
