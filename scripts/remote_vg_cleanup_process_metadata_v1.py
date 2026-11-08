"""Read-only metadata diagnosis of unreadable processes, without argument values."""
from pathlib import Path
from datetime import datetime, timezone
import json,os
rows=[]
pids=[]
for item in Path('/proc').iterdir():
 if not item.name.isdigit() or int(item.name)==os.getpid():continue
 try:
  if item.stat().st_uid!=os.getuid():continue
  try:list((item/'fd').iterdir())
  except PermissionError:pids.append(int(item.name))
 except OSError:pass
for pid in pids:
 p=Path('/proc')/str(pid);row={'pid':pid,'exists':p.exists()}
 if p.exists():
  for target in ['cwd','exe']:
   try:row[target]=os.readlink(p/target)
   except OSError as exc:row[target+'_error']=type(exc).__name__
  try:
   status={k:v.strip() for k,v in (line.split(':',1) for line in (p/'status').read_text().splitlines() if ':' in line)}
   row['status']={k:status.get(k) for k in ['Name','State','Pid','PPid','Uid','Gid','TracerPid','NSpid','NoNewPrivs','Seccomp']}
   parent=Path('/proc')/status.get('PPid','0')/'comm'
   try:row['parent_name']=parent.read_text().strip()
   except OSError:pass
  except OSError as exc:row['status_error']=type(exc).__name__
  try:
   args=(p/'cmdline').read_bytes().split(b'\0');row['argument_count']=len([a for a in args if a]);row['executable_argument_basename']=Path(args[0].decode(errors='replace')).name
   row['known_tokens']=[x for x in ['sshd','sftp-server','docker','containerd','remote_vg_method.py','rsg-former','python','manage.py','jupyter','gunicorn','node'] if any(x.encode() in a for a in args)]
  except OSError as exc:row['cmdline_error']=type(exc).__name__
 rows.append(row)
print(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),'current_uid':os.getuid(),'read_only':True,'processes':rows,'deletion_performed':False}))
