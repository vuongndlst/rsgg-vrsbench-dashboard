"""Account-owned storage metadata audit only; no file contents/models/deletion/server writes."""
from pathlib import Path
from datetime import datetime,timezone
import json,os,resource,signal,subprocess,shutil,time
start=time.monotonic();os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});os.nice(10);resource.setrlimit(resource.RLIMIT_AS,(512*1024**2,512*1024**2));resource.setrlimit(resource.RLIMIT_CPU,(70,70));signal.alarm(85)
home=Path.home();root=home/'rsg-former'
def alias(p):return str(p).replace(str(home),'${REMOTE_HOME}')
def sizes(path,depth,seconds):
 if not path.exists() or path.is_symlink():return {'present':path.exists(),'symlink':path.is_symlink()}
 try:r=subprocess.run(['du','-x','-B1','--max-depth='+str(depth),'--exclude=.ssh','--exclude=.aws','--exclude=.git','--',str(path)],capture_output=True,text=True,timeout=seconds)
 except subprocess.TimeoutExpired:return {'present':True,'complete':False,'reason':'time_cap'}
 rows=[]
 for line in r.stdout.splitlines():
  n,sep,p=line.partition('\t')
  if sep and n.isdigit():rows.append({'path':alias(p),'allocated_bytes':int(n)})
 return {'present':True,'complete':r.returncode==0,'error_lines':len(r.stderr.splitlines()),'rows':sorted(rows,key=lambda x:-x['allocated_bytes'])[:100]}
d=shutil.disk_usage(root);out={'utc':datetime.now(timezone.utc).isoformat(),'read_only':True,'server_writes':0,'files_deleted':0,'disk_total_bytes':d.total,'disk_used_bytes':d.used,'disk_free_bytes':d.free,'home':sizes(home,2,35),'details':{},'own_processes':[],'open_deleted_files':[],'fd_permission_errors':0}
for rel,depth in [('rsg-former/artifacts',2),('etechs',3),('.cache/huggingface',3),('.cache/pip',1),('shared-assets',2)]:
 if time.monotonic()-start>65:break
 out['details'][rel]=sizes(home/rel,depth,12)
seen=set()
for p in Path('/proc').iterdir():
 if not p.name.isdigit() or int(p.name)==os.getpid():continue
 try:
  if p.stat().st_uid!=os.getuid():continue
  cwd=os.readlink(p/'cwd');name=(p/'comm').read_text().strip();fstat=(p/'stat').read_text().split();cmd=(p/'cmdline').read_bytes().split(b'\x00');scripts=[Path(a.decode(errors='replace')).name for a in cmd if a.endswith(b'.py')]
  out['own_processes'].append({'pid':int(p.name),'name':name,'cwd':alias(cwd),'script_names':scripts,'start_ticks':fstat[21]})
  for fd in (p/'fd').iterdir():
   try:
    target=os.readlink(fd)
    if not target.endswith(' (deleted)'):continue
    s=fd.stat();key=(s.st_dev,s.st_ino)
    if key in seen:continue
    seen.add(key);out['open_deleted_files'].append({'pid':int(p.name),'path':alias(target),'allocated_bytes':s.st_blocks*512,'logical_bytes':s.st_size})
   except OSError:pass
 except OSError:out['fd_permission_errors']+=1
out['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
out['wall_seconds']=time.monotonic()-start;print(json.dumps(out))
