"""Read-only project/cache size inventory; no deletion, image reads or model runs."""
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict
import heapq,json,os,resource,shutil,signal,time

os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
resource.setrlimit(resource.RLIMIT_AS,(512*1024**2,512*1024**2))
resource.setrlimit(resource.RLIMIT_CPU,(75,75));signal.alarm(100)
home=Path.home();root=home/'rsg-former';start=time.monotonic()

def scan(base):
 groups=defaultdict(int);largest=[];seen=set();files=0;symlinks=0;errors=0;complete=True
 if not base.is_dir() or base.is_symlink():return {'present':base.exists(),'symlink':base.is_symlink()}
 for folder,dirs,names in os.walk(base,followlinks=False):
  dirs[:]=[d for d in dirs if not (Path(folder)/d).is_symlink()]
  if time.monotonic()-start>75:complete=False;break
  for name in names:
   p=Path(folder)/name
   try:
    s=p.lstat()
    if p.is_symlink():symlinks+=1;continue
    key=(s.st_dev,s.st_ino)
    if key in seen:continue
    seen.add(key);files+=1;size=s.st_blocks*512;rel=p.relative_to(base).as_posix();parts=Path(rel).parts
    for level in range(1,min(3,len(parts)-1)+1):groups['/'.join(parts[:level])]+=size
    if size>=32*1024**2:
     row=(size,rel,s.st_size,s.st_nlink,int(s.st_mtime),s.st_dev,s.st_ino)
     if len(largest)<60:heapq.heappush(largest,row)
     elif row>largest[0]:heapq.heapreplace(largest,row)
   except OSError:errors+=1
 return {'present':True,'complete':complete,'unique_files':files,'skipped_symlinks':symlinks,'errors':errors,
         'allocated_bytes':sum(v for k,v in groups.items() if '/' not in k),
         'groups':sorted([{'path':k,'allocated_bytes':v} for k,v in groups.items()],key=lambda x:-x['allocated_bytes'])[:65],
         'largest_files':[{'path':v[1],'allocated_bytes':v[0],'logical_bytes':v[2],'links':v[3],'mtime_unix':v[4],'device':v[5],'inode':v[6]} for v in sorted(largest,reverse=True)]}

out={'utc':datetime.now(timezone.utc).isoformat(),'read_only':True,'disk_free_bytes':shutil.disk_usage(root).free,
     'project':scan(root),'caches':{},'own_project_processes':[]}
for rel in ['.cache/pip','.cache/uv','.cache/torch_extensions']:
 out['caches'][rel]=scan(home/rel)
for p in Path('/proc').iterdir():
 if not p.name.isdigit() or int(p.name)==os.getpid():continue
 try:
  if p.stat().st_uid!=os.getuid():continue
  cwd=os.readlink(p/'cwd');cmd=(p/'cmdline').read_bytes()
  if cwd==str(root) or cwd.startswith(str(root)+'/') or str(root).encode() in cmd:
   out['own_project_processes'].append({'pid':int(p.name),'project_cwd':cwd==str(root) or cwd.startswith(str(root)+'/'),'known_job_tokens':[t for t in ['remote_vg_v2_time_extension_v1.py','remote_vg_component_control_test_v1.py','remote_vg_internal_phash_batch_v1.py'] if t.encode() in cmd]})
 except OSError:pass
out['wall_seconds']=time.monotonic()-start
print(json.dumps(out))
