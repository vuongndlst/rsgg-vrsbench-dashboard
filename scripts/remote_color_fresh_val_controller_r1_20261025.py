"""Sequential finite color validation cache and blind packet; no TEST/caption stage."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,subprocess,sys
root=Path.home()/'rsg-former';report=root/'reports/VG_COLOR_FRESH_VAL_CONTROLLER_R1_20261025.json'
assert not report.exists()
paths=['scripts/remote_vg_color_fresh_val_execution_r1_20261025.py','scripts/remote_color_fresh_val_review_packet_r1_20261025.py']
hashes={s:hashlib.sha256((root/s).read_bytes()).hexdigest() for s in paths};outcomes=[]
for phase in paths:
 for path,digest in hashes.items():assert hashlib.sha256((root/path).read_bytes()).hexdigest()==digest
 with (root/('reports/'+Path(phase).stem+'.stdout.log')).open('x') as out,(root/('reports/'+Path(phase).stem+'.stderr.log')).open('x') as err:
  child=subprocess.Popen([sys.executable,'-B',str(root/phase)],cwd=root,env=dict(os.environ),stdin=subprocess.DEVNULL,stdout=out,stderr=err)
  report.write_text(json.dumps({'status':'RUNNING','utc':datetime.now(timezone.utc).isoformat(),'phase':phase,'child_pid':child.pid,'source_sha256':hashes,'TEST_access_allowed':False},indent=2))
  code=child.wait();outcomes.append({'phase':phase,'returncode':code})
 if code:break
record={'status':'COMPLETE_HUMAN_COLOR_REVIEW_PENDING' if all(r['returncode']==0 for r in outcomes) else 'FAILED_NO_RETRY',
 'utc':datetime.now(timezone.utc).isoformat(),'outcomes':outcomes,'source_sha256':hashes,'TEST_accessed':False,'caption_generated':False}
report.write_text(json.dumps(record,indent=2));print(json.dumps(record))
