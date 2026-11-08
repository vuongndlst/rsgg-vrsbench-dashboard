"""One exclusive launch of the new TRAIN/VAL-only server caption cache phase."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,shutil,subprocess,sys
root=Path.home()/'rsg-former';lock=root/'reports/VG_CAPTION_CACHE_LAUNCH_20261025.json'
assert not lock.exists() and not (root/'artifacts/vg_caption_execution_v1_20261025').exists()
assert shutil.disk_usage(root).free>=17*1024**3
worker=root/'scripts/remote_vg_caption_cache_prepare_v1.py'
record={'status':'LAUNCHING','utc':datetime.now(timezone.utc).isoformat(),'worker_sha256':hashlib.sha256(worker.read_bytes()).hexdigest(),
 'phase':'vg_caption_execution_v1_20261025','test_access_allowed':False,'training_allowed':False}
with lock.open('x') as f:json.dump(record,f)
env=dict(os.environ);env.update(PYTHONDONTWRITEBYTECODE='1',CUDA_CACHE_DISABLE='1')
with (root/'reports/VG_CAPTION_CACHE_WORKER_20261025.stdout.log').open('x') as out,(root/'reports/VG_CAPTION_CACHE_WORKER_20261025.stderr.log').open('x') as err:
 p=subprocess.Popen([sys.executable,'-B',str(worker)],cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
record.update(status='LAUNCHED',pid=p.pid);lock.write_text(json.dumps(record,indent=2));print(json.dumps(record))
