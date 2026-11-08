from pathlib import Path
from datetime import datetime,timezone
import os,json,hashlib,resource,signal,shutil
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});resource.setrlimit(resource.RLIMIT_AS,(512*1024**2,512*1024**2));signal.alarm(60)
root=(Path.home()/'rsg-former').resolve();cache=root/'artifacts/vg_method_v2/features'
def consumers():
 blockers=[];unreadable=[];services=[]
 for p in Path('/proc').iterdir():
  if not p.name.isdigit() or int(p.name)==os.getpid():continue
  try:
   if p.stat().st_uid!=os.getuid():continue
   status=(p/'status').read_text();fields={k:v.strip() for k,v in (line.split(':',1) for line in status.splitlines() if ':' in line)}
   if fields['State'].startswith(('Z','X')):continue
   cmd=(p/'cmdline').read_bytes()
   if not cmd:continue
   if any(s.encode() in cmd for s in ['remote_vg_v2_pipeline.py','remote_vg_v2_speed_pipeline.py','remote_vg_v2_time_extension_v1.py']) or str(cache).encode() in cmd:blockers.append({'pid':int(p.name),'reason':'legacy_V2_training_or_cache_argument'})
   parent_name=(Path('/proc')/fields['PPid']/'comm').read_text().strip();args=[a for a in cmd.split(b'\0') if a];program=Path(args[0].decode(errors='replace')).name
   daemon=(len(args)==1 and ((fields['Name']=='(sd-pam)' and program=='(sd-pam)' and parent_name=='systemd') or (fields['Name']=='sshd' and program.startswith('sshd: ') and parent_name=='sshd') or (fields['Name']=='sftp-server' and program=='sftp-server' and parent_name=='sshd')))
   if daemon:services.append({'pid':int(p.name),'name':fields['Name'],'parent_name':parent_name});continue
   paths=[]
   try:
    for fd in (p/'fd').iterdir():
     try:paths.append(os.readlink(fd))
     except FileNotFoundError:pass
   except PermissionError:unreadable.append({'pid':int(p.name),'surface':'fd','name':fields['Name']})
   try:paths.extend(line.split(maxsplit=5)[-1] for line in (p/'maps').read_text().splitlines() if len(line.split(maxsplit=5))==6)
   except PermissionError:unreadable.append({'pid':int(p.name),'surface':'maps','name':fields['Name']})
   if any(x==str(cache) or x.startswith(str(cache)+'/') for x in paths):blockers.append({'pid':int(p.name),'reason':'open_or_mapped_cache'})
  except (FileNotFoundError,ProcessLookupError):pass
 return blockers,unreadable,services
blockers,unreadable,services=consumers()
raw=(cache/'manifest.json').read_bytes();m=json.loads(raw);entries={e['path']:e for es in m['splits'].values() for e in es}
assert hashlib.sha256(raw).hexdigest()=='f978799bbd19047b6a74f291aa816f43052251f9b84751f633e8ddf04622abf1'
assert len(entries)==981 and {p.name for p in cache.iterdir()}==set(entries)|{'manifest.json'}
assert not (root/'artifacts/maintenance/vg_v2_cache_offload_20261101').exists()
assert not (root/'artifacts/vg_focus_repair_r1_20261030').exists()
assert all((cache/n).stat().st_size==e['bytes'] for n,e in entries.items())
print(json.dumps({'status':'NOOP_OLD_FAILURE_CONFIRMED_CACHE981_INTACT','utc':datetime.now(timezone.utc).isoformat(),'server_writes':0,'server_files_deleted':0,'cache_file_count':981,'manifest_sha256':hashlib.sha256(raw).hexdigest(),'original_retirement_namespace_absent':True,'R1_namespace_absent':True,'own_consumer_blockers':blockers,'unreadable_application_processes':unreadable,'identified_platform_roles':services,'disk_free_bytes':shutil.disk_usage(root).free}))
