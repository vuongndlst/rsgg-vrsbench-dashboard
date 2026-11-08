"""Read-only, bounded GPU PID -> cwd/entrypoint/parent attribution for human request."""
from pathlib import Path
from datetime import datetime,timezone
import csv,json,os,re,resource,signal,subprocess,time
START=time.time();signal.alarm(45);os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});resource.setrlimit(resource.RLIMIT_AS,(256*1024**2,256*1024**2))
def run(args):
 r=subprocess.run(args,capture_output=True,text=True,timeout=12);return {'returncode':r.returncode,'stdout':r.stdout.strip(),'stderr':r.stderr.strip()[:300]}
def redact(args):
 out=[];hide=False
 for arg in args:
  if hide:out.append('<REDACTED>');hide=False;continue
  if re.search(r'(?i)(password|passwd|api[-_]?key|secret|token|authorization|credential)',arg):
   if '=' in arg:out.append(arg.split('=',1)[0]+'=<REDACTED>')
   elif arg.startswith('-'):out.append(arg);hide=True
   else:out.append('<REDACTED_SENSITIVE_ARGUMENT>')
  else:out.append(re.sub(r'(?i)(https?://)[^/@\s]+:[^/@\s]+@',r'\1<REDACTED>@',arg))
 return out

def process(pid,parents=False):
 base=Path('/proc')/str(pid);out={'pid':pid};errors={}
 try:
  data=(base/'stat').read_text().rsplit(')',1)[1].split();out.update(state=data[0],ppid=int(data[1]),start_ticks=data[19])
 except (OSError,IndexError,ValueError) as e:out['proc_metadata_unavailable']=type(e).__name__;return out
 try:out['uid']=int(next(s for s in (base/'status').read_text().splitlines() if s.startswith('Uid:')).split()[1])
 except (OSError,StopIteration,ValueError) as e:errors['uid']=type(e).__name__
 for key in ['cwd','exe']:
  try:out[key]=os.readlink(base/key)
  except OSError as e:errors[key]=type(e).__name__
 try:
  argv=[s.decode('utf8','replace') for s in (base/'cmdline').read_bytes().split(b'\0') if s]
  out['argv_redacted']=redact(argv)
 except OSError as e:errors['argv']=type(e).__name__
 if not parents:
  fds={}
  for fd in ['0','1','2']:
   try:fds[fd]=os.readlink(base/'fd'/fd)
   except OSError:pass
  out['stdio_targets']=fds
  chain=[];seen={pid};current=out['ppid']
  for _ in range(4):
   if current<=1 or current in seen:break
   seen.add(current);row=process(current,True);chain.append(row);current=row.get('ppid',0)
  out['parent_chain']=chain
 try:out['pid_identity_unchanged_during_read']=(base/'stat').read_text().rsplit(')',1)[1].split()[19]==out['start_ticks']
 except OSError:out['pid_identity_unchanged_during_read']=False
 if errors:out['permission_or_read_errors']=errors
 return out
fields=['index','uuid','name','memory.total','memory.used','memory.free','utilization.gpu','utilization.memory']
g=run(['nvidia-smi','--query-gpu='+','.join(fields),'--format=csv,noheader,nounits'])
gpus=[dict(zip(fields,[v.strip() for v in row])) for row in csv.reader(g['stdout'].splitlines())] if g['returncode']==0 else []
c=run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory','--format=csv,noheader,nounits'])
processes=[]
if c['returncode']==0:
 for row in csv.reader(c['stdout'].splitlines()):
  if len(row)!=4:continue
  uuid,pid,name,memory=[v.strip() for v in row]
  if not pid.isdigit():continue
  processes.append({'gpu_uuid':uuid,'pid':int(pid),'driver_process_name':name,'used_gpu_memory_MiB':memory,'proc':process(int(pid))})
pmon=run(['nvidia-smi','pmon','-c','1','-s','um'])
print(json.dumps({'status':'READONLY_GPU_PROCESS_PROJECT_EVIDENCE_SNAPSHOT','utc':datetime.now(timezone.utc).isoformat(),'gpus':gpus,'gpu_query_returncode':g['returncode'],'compute_query_returncode':c['returncode'],'compute_processes':processes,'process_activity_one_sample':pmon['stdout'],'process_activity_returncode':pmon['returncode'],
 'limits':['Memory is driver-reported allocated process VRAM, not torch tensor/caching breakdown.','One utilization sample cannot establish idle/leak or authorize killing jobs.','Project attribution requires entrypoint/cwd/parent evidence; unreadable proc or shared service is unresolved.','Device memory includes driver/context/graphics overhead not attributed by compute-process sum.','No environment variables, secrets, job logs or model/data contents read.'],
 'processes_signalled_or_modified':False,'GPU_allocations_created':False,'server_files_written':False,'wall_seconds':time.time()-START}))
