"""One launch verification for the new audit; separate from VG monitor."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json
root=Path.home()/'rsg-former'
phase=root/'artifacts/caption_readiness/oi_stream_phase_v1'
pid=int((phase/'pipeline.pid').read_text())
cmd=Path('/proc')/str(pid)/'cmdline'
out={'utc':datetime.now(timezone.utc).isoformat(), 'pid':pid,
     'active':cmd.exists() and b'remote_oi_stream_pipeline.py' in cmd.read_bytes(),
     'preregistration_sha256':hashlib.sha256((phase/'preregistration.json').read_bytes()).hexdigest(),
     'vg_job_checked_again':False, 'vg_job_modified':False}
for filename,key in [('OI_STREAM_FINGERPRINT_PROGRESS.json','progress'),('OI_STREAM_PHASE_RESULT.json','result')]:
    path=root/'reports'/filename
    if path.exists(): out[key]=json.loads(path.read_text())
if not out['active']:
    with (phase/'pipeline.log').open('rb') as stream:
        stream.seek(max(0,(phase/'pipeline.log').stat().st_size-3000))
        out['log_tail']=stream.read().decode(errors='replace')
(root/'reports/OI_STREAM_LAUNCH_CHECK.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out))
