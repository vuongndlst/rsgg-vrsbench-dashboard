"""Owned-process bounded color->Qwen->corpus score pilot; no automatic retries/TEST."""
from pathlib import Path
import json,os,signal,shutil,subprocess,sys,time,traceback,zipfile
ROOT=Path.home()/'rsg-former';PIPE=ROOT/'artifacts/vg_caption_pilot100_v1_20261026';REPORT=ROOT/'reports/VG_CAPTION_PILOT100_CONTROLLER_20261026.json'
os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:4]))
from rsgformer.vg_color_execution_common_v1 import sha,save,line,now,artifact_bytes

def rss_tree(pid):
 rows={}
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:
   t=(p/'status').read_text().splitlines();parent=int(next(v for v in t if v.startswith('PPid:')).split()[1]);rss=int(next((v for v in t if v.startswith('VmRSS:')),'VmRSS: 0').split()[1])*1024
   rows[int(p.name)]=(parent,rss)
  except (OSError,ValueError,StopIteration):continue
 ids={pid}
 while True:
  more={i for i,(parent,rss) in rows.items() if parent in ids}
  if more<=ids:break
  ids|=more
 return sum(rows.get(i,(0,0))[1] for i in ids),ids

def source_check(lock):
 for p,h in lock['source_sha256'].items():assert sha(ROOT/p)==h,('locked_source_drift',p)

def run_stage(name,script,gpu,rss,wall,lock):
 source_check(lock);assert not (PIPE/name).exists(),'no_duplicate_phase'
 assert shutil.disk_usage(ROOT).free>=17*1024**3 and artifact_bytes(ROOT/'artifacts')+1024**3<=40*1024**3
 if gpu:
  free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0])*1024**2
  assert free>=(38 if gpu==36 else 8)*1024**3,'GPU_admission'
 mem=int(next(v for v in Path('/proc/meminfo').read_text().splitlines() if v.startswith('MemAvailable:')).split()[1])*1024
 assert mem>=(rss+2)*1024**3,'host_RAM_admission'
 env=dict(os.environ);env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUDA_CACHE_DISABLE='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
 start=time.monotonic();peak_rss=peak_gpu=0;last_storage=-60;reason=None
 with (PIPE/(name+'.stdout.log')).open('x') as out,(PIPE/(name+'.stderr.log')).open('x') as err:
  child=subprocess.Popen([sys.executable,'-B',str(ROOT/'scripts'/script)],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
  state={'status':'RUNNING','stage':name,'controller_pid':os.getpid(),'child_pid':child.pid,'child_start_ticks':Path('/proc',str(child.pid),'stat').read_text().split()[21],
   'utc':now(),'own_GPU_cap_GiB':gpu,'own_tree_RSS_cap_GiB':rss,'wall_seconds':wall,'TEST_accessed':False}
  save(PIPE/'status.json',state);save(REPORT,state)
  while child.poll() is None:
   used,ids=rss_tree(child.pid);peak_rss=max(peak_rss,used)
   result=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=8)
   if result.returncode:reason='GPU_observation_failed';devices={}
   else:devices={int(v.split(',')[0]):int(v.split(',')[1])*1024**2 for v in result.stdout.splitlines() if v.strip()}
   own=sum(devices.get(i,0) for i in ids);peak_gpu=max(peak_gpu,own)
   if used>rss*1024**3:reason='owned_tree_RSS_cap'
   elif own>gpu*1024**3:reason='owned_GPU_cap'
   elif time.monotonic()-start>wall:reason='stage_wall_cap'
   elif time.time()>=1794670321.2414355:reason='overall_authorized_deadline'
   elif (PIPE/'STOP').exists():reason='owned_STOP'
   if time.monotonic()-start-last_storage>=60:
    if shutil.disk_usage(ROOT).free<16*1024**3:reason='disk_reserve16GiB'
    elif artifact_bytes(ROOT/'artifacts')>40*1024**3:reason='aggregate_artifacts40GiB'
    elif artifact_bytes(PIPE)>1024**3:reason='pilot_phase1GiB'
    last_storage=time.monotonic()-start
   if reason:os.killpg(child.pid,signal.SIGTERM);break
   time.sleep(3)
  try:child.wait(timeout=15)
  except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
 state.update(status='COMPLETE' if child.returncode==0 and not reason else 'FAILED',utc=now(),returncode=child.returncode,guard_reason=reason,
  seconds=time.monotonic()-start,peak_owned_tree_RSS_bytes=peak_rss,peak_owned_GPU_bytes=peak_gpu)
 save(PIPE/(name+'_outcome.json'),state);assert state['status']=='COMPLETE',(name,reason,child.returncode)
 source_check(lock);assert (PIPE/name/'complete.json').exists();return state

def main():
 assert not (PIPE/'started.json').exists();save(PIPE/'started.json',{'utc':now(),'pid':os.getpid()})
 lock=json.loads((PIPE/'preregistration.json').read_text());outcomes=[]
 for name,script,gpu,rss,wall in [
  ('prepare','remote_vg_caption_pilot100_prepare_v1_20261026.py',8,4,3600),
  ('generation','remote_vg_caption_pilot100_generate_v1_20261026.py',36,12,7200),
  ('metrics','remote_vg_caption_pilot100_score_v1_20261026.py',0,4,2700)]:outcomes.append(run_stage(name,script,gpu,rss,wall,lock))
 result={'status':'PILOT100_CAPTIONS_AND_ALL12_CONFIG_METRICS_COMPLETE','utc':now(),'controller_pid':os.getpid(),'outcomes':outcomes,
  'preregistration_sha256':sha(PIPE/'preregistration.json'),'metrics_completion_sha256':sha(PIPE/'metrics/complete.json'),'TEST_accessed':False,
  'caption_TEST_automatically_opened':False,'human_color_precision_gate_passed':False}
 save(PIPE/'stages_complete.json',result)
 files=[p for p in PIPE.rglob('*') if p.is_file() and p.suffix not in ['.log'] and p.name not in ['status.json']]
 assert sum(p.stat().st_size for p in files)<64*1024**2
 archive=PIPE/'pilot100_results.zip'
 with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
  for p in files:z.write(p,p.relative_to(PIPE).as_posix())
 result.update(archive=str(archive.relative_to(ROOT)),archive_sha256=sha(archive),archive_bytes=archive.stat().st_size)
 save(PIPE/'complete.json',result);save(PIPE/'status.json',result);save(REPORT,result);print(line(result),flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:
  state=json.loads((PIPE/'status.json').read_text()) if (PIPE/'status.json').exists() else {}
  state.update(status='FAILED_NO_AUTOMATIC_RETRY',utc=now(),reason=str(e),TEST_accessed=False)
  save(PIPE/'failure.json',state);save(PIPE/'status.json',state);save(REPORT,state);traceback.print_exc();raise
