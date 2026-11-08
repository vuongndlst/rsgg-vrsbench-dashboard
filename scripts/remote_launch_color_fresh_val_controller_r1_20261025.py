"""Exclusive fresh VAL color-cache launch after locked membership and resource admission."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,subprocess,sys,shutil
root=Path.home()/'rsg-former';phase=root/'artifacts/color_fresh_val_execution_r1_20261025';report=root/'reports/VG_COLOR_FRESH_VAL_LAUNCH_R1_20261025.json'
assert not phase.exists() and not report.exists()
assert shutil.disk_usage(root).free>=18*1024**3
free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0]);assert free>=8*1024
worker=root/'scripts/remote_color_fresh_val_controller_r1_20261025.py'
record={'utc':datetime.now(timezone.utc).isoformat(),'status':'LAUNCHING','worker_sha256':hashlib.sha256(worker.read_bytes()).hexdigest(),'TEST_access_allowed':False,'caption_launch_allowed':False}
with report.open('x') as f:json.dump(record,f)
with (root/'reports/VG_COLOR_FRESH_VAL_R1_20261025.stdout.log').open('x') as out,(root/'reports/VG_COLOR_FRESH_VAL_R1_20261025.stderr.log').open('x') as err:
 p=subprocess.Popen([sys.executable,'-B',str(worker)],cwd=root,env=dict(os.environ),stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
record.update(status='LAUNCHED',pid=p.pid);report.write_text(json.dumps(record,indent=2));print(json.dumps(record))
