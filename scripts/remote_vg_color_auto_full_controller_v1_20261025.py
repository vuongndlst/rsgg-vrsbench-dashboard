"""Immutable finite stage controller. No retries, tuning, caption or training launch."""
from pathlib import Path
import json,os,subprocess,sys,time,traceback
ROOT=Path.home()/'rsg-former';ART=ROOT/'artifacts/vg_color_auto_full_pipeline_v1_20261025'
os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:4]))
from rsgformer.vg_color_execution_common_v1 import sha,save,line,now
REPORT=ROOT/'reports/VG_COLOR_AUTO_FULL_PIPELINE_V1_20261025.json'

def main():
 lock=json.loads((ART/'preregistration.json').read_text());assert not (ART/'started.json').exists()
 save(ART/'started.json',{'utc':now(),'pid':os.getpid()})
 state={'utc':now(),'status':'RUNNING','controller_pid':os.getpid(),'preregistration_sha256':sha(ART/'preregistration.json'),'completed_stages':[],'human_precision_gate_passed':False}
 stages=[('VAL_PARITY','remote_vg_color_fast_val_replay_v1_20261025.py','vg_color_fast_val_replay_v1_20261025',3600),
  ('FULL_TEST_COLOR','remote_vg_color_full_test_auto_v1_20261025.py','vg_color_full_test_auto_v1_20261025',82800),
  ('TEST_ATTRIBUTE_DIAGNOSTIC','remote_vg_color_auto_test_diagnostic_v1_20261025.py','vg_color_auto_test_diagnostic_v1_20261025',3600)]
 for stage,script,folder,limit in stages:
  assert time.time()<1794670321.2414355 and not (ART/'STOP').exists(),'controller_deadline_or_STOP'
  assert not (ROOT/'artifacts'/folder).exists(),'no_blind_retry'
  for p,h in lock['source_sha256'].items():assert sha(ROOT/p)==h,('source_drift',p)
  assert sha(ROOT/'configs/vg_color_automated_full_test_v1_20261025.json')==lock['config_sha256']
  state.update(utc=now(),stage=stage,status='RUNNING')
  with (ART/(stage+'.stdout.log')).open('x') as out,(ART/(stage+'.stderr.log')).open('x') as err:
   child=subprocess.Popen([sys.executable,'-B',str(ROOT/'scripts'/script)],cwd=ROOT,env=dict(os.environ),stdin=subprocess.DEVNULL,stdout=out,stderr=err)
   state['child_pid']=child.pid;save(REPORT,state);save(ART/'status.json',state)
   try:code=child.wait(timeout=limit+120)
   except subprocess.TimeoutExpired:
    # Only this controller's own identified child is stopped. Never restart partial TEST.
    child.terminate()
    try:child.wait(timeout=30)
    except subprocess.TimeoutExpired:child.kill();child.wait()
    raise RuntimeError('own_child_wall_timeout_'+stage)
  assert code==0,('child_failed',stage,code)
  result_path=ROOT/'artifacts'/folder/'complete.json';result=json.loads(result_path.read_text())
  if stage=='VAL_PARITY':assert result['status']=='FAST_VAL_REPLAY_PARITY_PASS'
  state['completed_stages'].append({'stage':stage,'completion_sha256':sha(result_path),'status':result['status']})
 state.update(utc=now(),status='COMPLETE_AUTOMATED_COLOR_TEST_NOT_HUMAN_PRECISION',stage='COMPLETE',child_pid=None)
 save(ART/'complete.json',state);save(ART/'status.json',state);save(REPORT,state);print(line(state),flush=True)

if __name__=='__main__':
 try:main()
 except Exception as error:
  state=json.loads((ART/'status.json').read_text()) if (ART/'status.json').exists() else {}
  state.update(utc=now(),status='BLOCKED_NO_AUTOMATIC_RETRY',error_type=type(error).__name__,reason=str(error),
   TEST_color_access_opened=(ROOT/'artifacts/vg_color_full_test_auto_v1_20261025/test_access.json').exists())
  save(ART/'failure.json',state);save(ART/'status.json',state);save(REPORT,state);traceback.print_exc();raise
