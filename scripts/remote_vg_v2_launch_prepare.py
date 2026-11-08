from pathlib import Path
import subprocess,os,json,time
root=Path.home()/'rsg-former';a=root/'artifacts/vg_method_v2';a.mkdir(exist_ok=True,parents=True)
pidfile=a/'prepare.pid';runner=root/'scripts/remote_vg_v2_prepare.py'
if pidfile.exists():
 p=int(pidfile.read_text());c=Path('/proc')/str(p)/'cmdline'
 if c.exists() and str(runner).encode() in c.read_bytes():print(json.dumps({'status':'already_running','pid':p}));raise SystemExit
env=dict(os.environ);env['PYTHONPATH']=str(root/'.tools/deps')+':'+str(root/'src');env['PYTHONDONTWRITEBYTECODE']='1'
with (a/'prepare.log').open('ab',buffering=0) as log:
 p=subprocess.Popen([str(root/'.venv/bin/python'),'-B',str(runner)],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
pidfile.write_text(str(p.pid));time.sleep(2);print(json.dumps({'status':'started' if p.poll() is None else 'early_exit','pid':p.pid,'log':'artifacts/vg_method_v2/prepare.log','test_scored':False}))
