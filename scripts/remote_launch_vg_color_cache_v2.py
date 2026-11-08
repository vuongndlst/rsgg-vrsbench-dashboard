"""Exclusive candidate-color cache launch after SG and Qwen pilot completion."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,subprocess,sys
root=Path.home()/'rsg-former';base=root/'artifacts/vg_caption_execution_v1_20261025'
assert (base/'base_cache_complete.json').exists() and not (base/'color_candidates_v2').exists()
lock=root/'reports/VG_COLOR_CACHE_V2_LAUNCH_20261025.json';assert not lock.exists()
worker=root/'scripts/remote_vg_caption_color_cache_v2.py'
record={'utc':datetime.now(timezone.utc).isoformat(),'status':'LAUNCHING','worker_sha256':hashlib.sha256(worker.read_bytes()).hexdigest(),'test_access':False}
with lock.open('x') as f:json.dump(record,f)
with (root/'reports/VG_COLOR_CACHE_V2_20261025.stdout.log').open('x') as out,(root/'reports/VG_COLOR_CACHE_V2_20261025.stderr.log').open('x') as err:
 p=subprocess.Popen([sys.executable,'-B',str(worker)],cwd=root,env=dict(os.environ),stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
record.update(status='LAUNCHED',pid=p.pid);lock.write_text(json.dumps(record,indent=2));print(json.dumps(record))
