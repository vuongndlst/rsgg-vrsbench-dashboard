"""Launch only v2, with separate locks/logs and inherited system packages unchanged."""
from pathlib import Path
import os,subprocess,json,time
root=Path.home()/'rsg-former';a=root/'artifacts/vg_method_v2';a.mkdir(parents=True,exist_ok=True)
runner=root/'scripts/remote_vg_v2_pipeline.py';pidfile=a/'pipeline.pid'
if (root/'reports/VG_V2_RESULT.json').exists():
 print(json.dumps({'status':'existing_result_do_not_restart','report':'reports/VG_V2_RESULT.json'}));raise SystemExit(1)
if pidfile.exists():
 pid=int(pidfile.read_text());c=Path('/proc')/str(pid)/'cmdline'
 if c.exists() and str(runner).encode() in c.read_bytes():print(json.dumps({'status':'already_running','pid':pid}));raise SystemExit
env=dict(os.environ);env.update(PYTHONPATH=':'.join(str(root/p) for p in ['.tools/vg_v2_deps','.tools/deps','src']),PYTHONDONTWRITEBYTECODE='1',CUDA_CACHE_DISABLE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
with (a/'pipeline.log').open('ab',buffering=0) as log:
 p=subprocess.Popen([str(root/'.venv/bin/python'),'-B',str(runner)],cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
pidfile.write_text(str(p.pid));time.sleep(3)
o={'status':'launched' if p.poll() is None else 'early_exit','pid':p.pid,'log':'artifacts/vg_method_v2/pipeline.log','test_scored':False,'phase':'awaiting assets, then gated train-only probe'}
(root/'reports/VG_V2_LAUNCH.json').write_text(json.dumps(o,indent=2));print(json.dumps(o))
