"""Exclusive bounded TRAIN engineering pilot, no caption or TEST launch."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,subprocess,sys,shutil
root=Path.home()/'rsg-former';phase=root/'artifacts/color_rules_v4_train_pilot_20261025';report=root/'reports/VG_COLOR_RULES_V4_TRAIN_PILOT_LAUNCH_20261025.json'
assert not phase.exists() and not report.exists()
assert shutil.disk_usage(root).free>=17*1024**3
free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0]);assert free>=8*1024
worker=root/'scripts/remote_vg_color_rules_v4_train_pilot_20261025.py'
record={'utc':datetime.now(timezone.utc).isoformat(),'status':'LAUNCHING','worker_sha256':hashlib.sha256(worker.read_bytes()).hexdigest(),'TEST_access_allowed':False,'caption_generator_launched':False}
with report.open('x') as f:json.dump(record,f)
with (root/'reports/VG_COLOR_RULES_V4_TRAIN_PILOT_20261025.stdout.log').open('x') as out,(root/'reports/VG_COLOR_RULES_V4_TRAIN_PILOT_20261025.stderr.log').open('x') as err:
 child=subprocess.Popen([sys.executable,'-B',str(worker)],cwd=root,env=dict(os.environ),stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
record.update(status='LAUNCHED',pid=child.pid);report.write_text(json.dumps(record,indent=2));print(json.dumps(record))
