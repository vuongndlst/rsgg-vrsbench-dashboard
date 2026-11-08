"""Read-only status for the single, explicitly amended component-test phase."""
from pathlib import Path
from datetime import datetime,timezone
import json
r=Path.home()/'rsg-former';a=r/'artifacts/vg_component_control_test_v1';out={'utc':datetime.now(timezone.utc).isoformat(),'phase_id':'vg_component_control_test_v1'}
out.update(phase_exists=a.exists(),test_opened=(a/'test_access.json').exists())
if (a/'preregistration.json').exists():
 import hashlib
 out['preregistration_sha256']=hashlib.sha256((a/'preregistration.json').read_bytes()).hexdigest()
if (a/'supersession.json').exists():out['supersession']=json.loads((a/'supersession.json').read_text())
for name in ['VG_COMPONENT_CONTROL_TEST_LAUNCH.json','VG_COMPONENT_CONTROL_TEST_PROGRESS.json','VG_COMPONENT_CONTROL_VALIDATION_GATE.json','VG_COMPONENT_CONTROL_TEST_RESULT.json']:
 p=r/'reports'/name
 if p.exists():out[name]=json.loads(p.read_text())
if (a/'pipeline.pid').exists():
 pid=int((a/'pipeline.pid').read_text());p=Path('/proc')/str(pid)/'cmdline';out['process']={'pid':pid,'active':p.exists() and b'remote_vg_component_control_test_v1.py' in p.read_bytes()}
for name in ['pipeline.stderr.log','pipeline.log']:
 p=a/name
 if p.exists():
  with p.open('rb') as f:f.seek(max(0,p.stat().st_size-1800));out[name]=f.read().decode(errors='replace')
print(json.dumps(out))
