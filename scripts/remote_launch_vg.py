"""Launch only the project-owned bounded VG experiment and return its PID."""
from pathlib import Path
import os,json,subprocess,time
ROOT=Path.home()/'rsg-former';folder=ROOT/'artifacts/vg_method_v1';folder.mkdir(parents=True,exist_ok=True)
tests=json.loads((ROOT/'reports/VG_METHOD_TESTS.json').read_text());probe=json.loads((ROOT/'reports/VG_METHOD_CONTRACT_PROBE.json').read_text())
assert tests['status']=='PASS' and probe['status']=='PASS' and not probe['validation_scored']
pidfile=folder/'pipeline.pid';runner=ROOT/'scripts/remote_vg_method.py'
if pidfile.exists():
    pid=int(pidfile.read_text());cmdline=Path(f'/proc/{pid}/cmdline')
    if cmdline.exists() and str(runner).encode() in cmdline.read_bytes():
        print(json.dumps({'status':'already_running','pid':pid}));raise SystemExit(0)
environment=dict(os.environ);environment['PYTHONPATH']=str(ROOT/'.tools/deps')+':'+str(ROOT/'src');environment['PYTHONDONTWRITEBYTECODE']='1';environment['CUDA_CACHE_DISABLE']='1'
log=open(folder/'pipeline.log','ab',buffering=0)
process=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-B',str(runner)],cwd=ROOT,env=environment,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
pidfile.write_text(str(process.pid));time.sleep(3)
out={'status':'launched' if process.poll() is None else 'early_exit','pid':process.pid,'returncode':process.poll(),'log':'artifacts/vg_method_v1/pipeline.log','started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'project_root':'${REMOTE_HOME}/rsg-former','test_scored':False}
(ROOT/'reports/VG_METHOD_LAUNCH.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
if process.poll() is not None:raise SystemExit(1)
