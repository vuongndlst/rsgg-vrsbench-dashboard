"""Exclusive launch of the authorized finite six-arm caption VAL experiment."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,subprocess,sys
root=Path.home()/'rsg-former';base=root/'artifacts/vg_caption_execution_v1_20261025'
lock=root/'reports/VG_CAPTION_DEV_CHECK_LAUNCH_20261025.json'
assert not lock.exists() and not (base/'dev_check_controller_v1').exists()
assert json.loads((root/'reports/VG_CAPTION_CONTRACTS_V1_20261025.json').read_text())['status']=='PROMPT_AND_NATIVE_METRIC_CONTRACT_PASS'
worker=root/'scripts/remote_vg_caption_dev_check_controller_v1.py'
record={'utc':datetime.now(timezone.utc).isoformat(),'status':'LAUNCHING','worker_sha256':hashlib.sha256(worker.read_bytes()).hexdigest(),'test_access_allowed':False}
with lock.open('x') as f:json.dump(record,f)
with (root/'reports/VG_CAPTION_DEV_CHECK_20261025.stdout.log').open('x') as out,(root/'reports/VG_CAPTION_DEV_CHECK_20261025.stderr.log').open('x') as err:
 p=subprocess.Popen([sys.executable,'-B',str(worker)],cwd=root,env=dict(os.environ),stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
record.update(status='LAUNCHED',pid=p.pid);lock.write_text(json.dumps(record,indent=2));print(json.dumps(record))
