"""Exclusive finite TRAIN speed/parity benchmark; caps unchanged, no TEST launch."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,shutil,subprocess,sys
root=Path.home()/'rsg-former';art=root/'artifacts/vg_color_speed_benchmark_v1_20261025';report=root/'reports/VG_COLOR_SPEED_BENCHMARK_LAUNCH_20261025.json'
assert not art.exists() and not report.exists()
assert shutil.disk_usage(root).free>=16*1024**3+128*1024**2
free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0]);assert free>=8*1024
old=json.loads((root/'reports/VG_COLOR_FRESH_VAL_CONTROLLER_R1_20261025.json').read_text());assert old['status']=='COMPLETE_HUMAN_COLOR_REVIEW_PENDING'
worker=root/'scripts/remote_vg_color_speed_benchmark_v1_20261025.py';record={'utc':datetime.now(timezone.utc).isoformat(),'status':'LAUNCHING','worker_sha256':hashlib.sha256(worker.read_bytes()).hexdigest(),'TEST_access_allowed':False}
with report.open('x') as f:json.dump(record,f)
with (root/'reports/VG_COLOR_SPEED_BENCHMARK_20261025.stdout.log').open('x') as out,(root/'reports/VG_COLOR_SPEED_BENCHMARK_20261025.stderr.log').open('x') as err:
 child=subprocess.Popen([sys.executable,'-B',str(worker)],cwd=root,env=dict(os.environ),stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
record.update(status='LAUNCHED',pid=child.pid);report.write_text(json.dumps(record,indent=2));print(json.dumps(record))
