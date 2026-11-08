"""Read-only admission and preserved run evidence for human-requested seed replication."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,resource,shutil,subprocess,time
ROOT=Path.home()/'rsg-former';ART=ROOT/'artifacts/vg_method_v2_speed_m2_v1'
initial_cpu_count=len(os.sched_getaffinity(0));os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});resource.setrlimit(resource.RLIMIT_AS,(384*1024**2,384*1024**2))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def process(pid):
 p=Path('/proc')/str(pid)
 try:
  cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace');s=(p/'stat').read_text().rsplit(')',1)[1].split()
  return {'pid':pid,'alive':s[0]!='Z','same_project_command':'rsg-former' in cmd or 'remote_vg_' in cmd,'command_basename':cmd.split('/')[-1][:160]}
 except (FileNotFoundError,ProcessLookupError):return {'pid':pid,'alive':False}
q=subprocess.run(['nvidia-smi','--query-gpu=index,name,memory.total,memory.free,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15)
gpu=[dict(zip(['index','name','total_mib','free_mib','utilization_percent'],[v.strip() for v in line.split(',')])) for line in q.stdout.splitlines()] if q.returncode==0 else []
mem={s.split(':')[0]:int(s.split(':')[1].split()[0])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.endswith('kB')}
runs=[]
for p in sorted(ART.glob('confirmation_*')):
 if not p.is_dir():continue
 report=p/'report.json';row=json.loads(report.read_text()) if report.exists() else {}
 files=[{'name':x.name,'bytes':x.stat().st_size} for x in p.iterdir() if x.is_file()]
 runs.append({'folder':p.name,'status':row.get('status'),'seed':row.get('seed'),'variant':row.get('variant'),'completed_epochs':len(row.get('history',[])),'selected_epoch':row.get('selected_epoch'),'selected_metrics':row.get('selected_metrics'),'initial_head_sha256':row.get('initial_head_sha256'),'checkpoint_sha256':row.get('checkpoint_sha256'),'report_sha256':sha(report) if report.exists() else None,'files':files})
stops=[]
for p in [ART/'STOP',ART/'time_extension_v1/STOP',ROOT/'artifacts/vg_component_control_test_v1/STOP']:
 if p.exists():stops.append({'relative':p.relative_to(ROOT).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size})
source={}
for name in ['src/rsgformer/vg_v2_runtime.py','src/rsgformer/vg_speedup.py','src/rsgformer/vg_speedup_schedule.py','scripts/remote_vg_v2_time_extension_v1.py','configs/vg_method_v2_time_extension_v1.json','reports/VG_TIME_EXTENSION_V1_PREREGISTRATION.json','artifacts/vg_method_v2/features/manifest.json']:
 p=ROOT/name;source[name]={'exists':p.is_file(),'sha256':sha(p) if p.is_file() else None}
free=shutil.disk_usage(ROOT).free;artifact_bytes=sum(x.stat().st_size for parent in [ROOT/'artifacts/vg_method_v2',ART,ROOT/'artifacts/vg_component_control_test_v1'] for x in parent.rglob('*') if x.is_file())
deadline=1794670321.2414355;remaining=deadline-time.time();epochs=[e['seconds'] for x in ART.glob('confirmation_*/report.json') for e in json.loads(x.read_text()).get('history',[]) if 'seconds' in e]
bound=max(epochs)*1.25 if epochs else None;training_required=26*bound if bound else None
out={'utc':datetime.now(timezone.utc).isoformat(),'read_only':True,'requested_variant':'context','requested_seeds':[2718,3141],'processes':[process(p) for p in [2993084,1051302]],'stop_ownership_preserved':stops,'runs':runs,'source_files':source,'resources':{'disk_free_bytes':free,'minimum_disk_free_bytes':16*1024**3,'new_phase_reserve_bytes':2*1024**3,'disk_admitted':free>=18*1024**3,'gpus':gpu,'gpu_admitted':bool(gpu) and int(gpu[0]['free_mib'])>=9*1024,'ram_available_bytes':mem.get('MemAvailable'),'cpu_available_before_probe_pinning':initial_cpu_count,'probe_cpu_affinity_count':1,'aggregate_accounted_artifact_bytes':artifact_bytes,'aggregate_limit_bytes':40*1024**3,'artifact_admitted':artifact_bytes+2*1024**3<=40*1024**3,'authorized_deadline_unix':deadline,'remaining_seconds':remaining,'completed_epoch_count':len(epochs),'max_completed_epoch_seconds':max(epochs) if epochs else None,'training_only_required_seconds_2seeds_13epochs_margin25pct':training_required,'time_admitted':training_required is not None and remaining>=training_required},'old_job_resumed':False,'test_accessed':False,'files_modified':False}
print(json.dumps(out))
