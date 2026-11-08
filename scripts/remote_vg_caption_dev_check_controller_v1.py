"""One finite sequential VAL experiment, owned-process guards; never opens TEST."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,signal,shutil,subprocess,sys,time,traceback
ROOT=Path.home()/'rsg-former';BASE=ROOT/'artifacts/vg_caption_execution_v1_20261025';ART=BASE/'dev_check_controller_v1'
SOURCES=['scripts/remote_vg_caption_dev_check_controller_v1.py','scripts/remote_vg_caption_generation_dev_v1.py',
 'scripts/remote_vg_caption_generation_check_v1.py','scripts/remote_vg_caption_score_dev_check_v1.py',
 'src/rsgformer/vg_caption_prompt_v1.py','src/rsgformer/vg_caption_graph_v1.py','src/rsgformer/qwen_streaming_v1.py',
 'src/rsgformer/caption_metrics_v2.py','src/rsgformer/caption_metric_runtime.py','configs/qwen_runtime.json']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def save(p,d):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,indent=2));t.replace(p)
def tracked_bytes():
 # Existing accounting scope is project artifacts, excluding external model/dataset assets.
 total=0;seen=set()
 for p in (ROOT/'artifacts').rglob('*'):
  if p.is_symlink() or not p.is_file():continue
  stat=p.stat();identity=(stat.st_dev,stat.st_ino)
  if identity not in seen:total+=stat.st_size;seen.add(identity)
 return total
def source_check(lock):
 for path,expected in lock.items():assert sha(ROOT/path)==expected,'locked_source_changed:'+path
def owned_tree_rss(pid):
 rows={}
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:
   lines=(p/'status').read_text().splitlines();ppid=int(next(x for x in lines if x.startswith('PPid:')).split()[1])
   rss=int(next((x for x in lines if x.startswith('VmRSS:')),'VmRSS: 0').split()[1])*1024
   rows[int(p.name)]=(ppid,rss)
  except (OSError,StopIteration,ValueError):continue
 ids={pid}
 while True:
  more={i for i,(parent,rss) in rows.items() if parent in ids}
  if more<=ids:break
  ids|=more
 return sum(rows.get(i,(0,0))[1] for i in ids),ids
def nvml_pids():
 q=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=8,check=True)
 return {int(line.split(',')[0]):int(line.split(',')[1])*1024**2 for line in q.stdout.splitlines() if line.strip()}
def run_stage(name,args,gpu,wall,lock):
 source_check(lock);assert shutil.disk_usage(ROOT).free>=17*1024**3,'disk_admission17GiB'
 assert tracked_bytes()<=40*1024**3,'aggregate_artifacts40GiB'
 assert time.time()<1794670321.2414355 and not (BASE/'STOP').exists()
 if gpu:
  free=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
  assert int(free.splitlines()[0])>=38*1024,'free_GPU38GiB_admission'
  mem=next(line for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
  assert int(mem.split()[1])*1024>=14*1024**3,'host_RAM14GiB_admission'
 start=time.monotonic();reason=None;peak_rss=peak_gpu=0;last_storage_check=-60
 env=dict(os.environ);env.update(PYTHONDONTWRITEBYTECODE='1',CUDA_CACHE_DISABLE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
  OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
 with (ART/(name+'.stdout.log')).open('x') as stdout,(ART/(name+'.stderr.log')).open('x') as stderr:
  p=subprocess.Popen([sys.executable,'-B']+args,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,start_new_session=True)
  stage={'stage':name,'pid':p.pid,'owned_process_group':p.pid,'start_ticks':Path('/proc',str(p.pid),'stat').read_text().split()[21],
   'started_utc':now(),'gpu_phase':gpu,'wall_cap_seconds':wall,'status':'RUNNING'}
  save(ART/'progress.json',stage)
  while p.poll() is None:
   rss,ids=owned_tree_rss(p.pid);peak_rss=max(peak_rss,rss)
   try:gpus=nvml_pids()
   except Exception as error:reason='watchdog_GPU_observation_failed:'+type(error).__name__;gpus={}
   used=sum(gpus.get(i,0) for i in ids);peak_gpu=max(peak_gpu,used)
   if rss>(12 if gpu else 4)*1024**3:reason='owned_tree_RSS_cap'
   elif used>(36 if gpu else 0)*1024**3:reason='owned_GPU_cap'
   elif time.monotonic()-start>wall:reason='stage_wall_cap'
   elif time.time()>=1794670321.2414355:reason='authorized_overall_deadline'
   elif (BASE/'STOP').exists():reason='owned_STOP'
   if time.monotonic()-start-last_storage_check>=60:
    if shutil.disk_usage(ROOT).free<16*1024**3:reason='disk_reserve16GiB'
    elif tracked_bytes()>40*1024**3:reason='aggregate_artifacts40GiB'
    elif sum(x.stat().st_size for x in BASE.rglob('*') if x.is_file())>8*1024**3:reason='phase_artifacts8GiB'
    last_storage_check=time.monotonic()-start
   if reason:
    os.killpg(p.pid,signal.SIGTERM);break
   time.sleep(2)
  try:p.wait(timeout=15)
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
 stage.update(status='COMPLETE' if p.returncode==0 and not reason else 'FAILED',returncode=p.returncode,guard_reason=reason,
  ended_utc=now(),peak_owned_tree_RSS_bytes=peak_rss,peak_owned_GPU_bytes=peak_gpu,seconds=time.monotonic()-start)
 save(ART/(name+'_outcome.json'),stage);save(ART/'progress.json',stage)
 assert stage['status']=='COMPLETE',name+':'+str(reason)+':exit'+str(p.returncode)
 source_check(lock)
 return stage
def main():
 assert not ART.exists(),'exclusive_controller_namespace_already_exists'
 contract=json.loads((ROOT/'reports/VG_CAPTION_CONTRACTS_V1_20261025.json').read_text())
 assert contract['status']=='PROMPT_AND_NATIVE_METRIC_CONTRACT_PASS'
 assert (BASE/'color_candidates_v2/complete.json').exists()
 ART.mkdir();os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:4]))
 lock={s:sha(ROOT/s) for s in SOURCES}
 save(ART/'preregistration.json',{'utc':now(),'source_sha256':lock,'authorization':'Human25Oct approved server cache then finite captions; Qwen36/32/12GiB CPU4 conditional headroom',
  'sequence':['dev_generation','dev_metrics_and_common_prompt_lock','heldout_check_generation','heldout_check_metrics'],
  'automatic_retry':False,'TEST_access_allowed':False,'all_six_VAL_arms_reported':True,'color_TEST_gate_passed':False,
  'overall_deadline_unix':1794670321.2414355,'aggregate_artifacts_GiB':40,'reserve_GiB':16})
 outcomes=[]
 for name,args,gpu,wall in [
  ('dev_generation',[str(ROOT/'scripts/remote_vg_caption_generation_dev_v1.py')],True,4*3600),
  ('dev_metrics',[str(ROOT/'scripts/remote_vg_caption_score_dev_check_v1.py'),'dev'],False,45*60),
  ('check_generation',[str(ROOT/'scripts/remote_vg_caption_generation_check_v1.py')],True,2*3600),
  ('check_metrics',[str(ROOT/'scripts/remote_vg_caption_score_dev_check_v1.py'),'check'],False,45*60)]:
  outcomes.append(run_stage(name,args,gpu,wall,lock))
 result={'status':'SIX_ARM_VAL_DEV_CHECK_PROGRAM_COMPLETE_TEST_STILL_LOCKED','utc':now(),'outcomes':outcomes,'test_accessed':False,'color_visual_gate_passed':False}
 save(ART/'complete.json',result);save(ROOT/'reports/VG_CAPTION_DEV_CHECK_CONTROLLER_20261025.json',result)
 print(json.dumps(result),flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:
  if ART.exists():save(ART/'failure.json',{'utc':now(),'error_type':type(e).__name__,'reason':str(e),'test_accessed':False})
  traceback.print_exc();raise
