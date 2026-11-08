"""Preserve the failed pre-training attempt before a separately checked relaunch."""
from pathlib import Path
import json,time,shutil
r=Path.home()/'rsg-former';a=r/'artifacts/vg_method_v2';p=a/'pipeline.pid'
if p.exists():
 pid=int(p.read_text());cmd=Path('/proc')/str(pid)/'cmdline'
 assert not (cmd.exists() and b'remote_vg_v2_pipeline.py' in cmd.read_bytes()),'cannot_edit_active_pipeline'
result=r/'reports/VG_V2_RESULT.json'
assert result.exists() and json.loads(result.read_text())['status']=='stopped_on_error'
assert not (a/'features/manifest.json').exists(),'not_a_preparation_only_failure'
d=a/'repairs'/time.strftime('%Y%m%dT%H%M%SZ',time.gmtime());d.mkdir(parents=True,exist_ok=False)
for name in ['VG_V2_RESULT.json','VG_V2_PROBE.json','VG_V2_PROGRESS.json']:
 source=r/'reports'/name
 if source.exists():shutil.copy2(source,d/name)
for name in ['pipeline.log','probe.log']:
 source=a/name
 if source.exists():shutil.copy2(source,d/name)
result.rename(d/'terminal_result_preserved.json')
(d/'REPAIR.json').write_text(json.dumps({'reason':'Technical NumPy/PyTorch checkpoint compatibility, before data cache or training','native_criterion_changed':False,'test_scored':False,'next_step':'train-only probe required; new source lock before training'},indent=2))
print(json.dumps({'status':'preparation_failure_archived','folder':str(d.relative_to(r)),'test_scored':False}))
