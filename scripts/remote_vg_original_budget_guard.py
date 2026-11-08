"""Retain original experiment's absolute wall deadline across audited resumes."""
from pathlib import Path
from datetime import datetime,timezone
import json,sys,time,hashlib
root=Path.home()/'rsg-former';folder=root/'artifacts/caption_readiness/vg_speedup_controller_v1'
protocol=json.loads((folder/'preregistration.json').read_text())
pid=int(sys.argv[1]);deadline=protocol['original_wall_deadline_unix']
while True:
    cmd=Path('/proc')/str(pid)/'cmdline'
    if not cmd.exists() or b'remote_vg_v2_pipeline.py' not in cmd.read_bytes():break
    if time.time()>=deadline:
        source='scripts/remote_vg_v2_pipeline.py'
        assert hashlib.sha256((root/source).read_bytes()).hexdigest()==protocol['source_hashes'][source]
        stop=root/'artifacts/vg_method_v2/STOP'
        if not stop.exists():
            with stop.open('x') as f:f.write('original_aggregate_240h_wall_budget')
        report={'utc':datetime.now(timezone.utc).isoformat(),'status':'ORIGINAL_WALL_BUDGET_STOP_REQUESTED','pid':pid,'deadline_unix':deadline,'other_process_signals':False}
        (root/'reports/VG_ORIGINAL_BUDGET_GUARD.json').write_text(json.dumps(report,indent=2))
        break
    time.sleep(min(60,max(.1,deadline-time.time())))
