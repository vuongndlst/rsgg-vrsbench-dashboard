"""User-authorized exact retirement of 480 V1 feature shards; no recursive deletion."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib,json,os,resource,shutil,signal,stat

os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
resource.setrlimit(resource.RLIMIT_AS,(512*1024**2,512*1024**2));signal.alarm(100)
root=(Path.home()/'rsg-former').resolve();cache=root/'artifacts/vg_method_v1/features'
assert cache.is_relative_to(root) and cache!=root and cache.resolve()==cache and not cache.is_symlink()
manifest_path=cache/'manifest.json';raw=manifest_path.read_bytes()
manifest_sha=hashlib.sha256(raw).hexdigest()
assert manifest_sha=='d39354aa61511c2bc3a02fa0b0115a1ad6473c2a48fb8ffcc4936043de95519e'
manifest=json.loads(raw);assert manifest['status']=='complete'
plan=json.loads((root/'reports/VG_V1_CACHE_CLEANUP_PREFLIGHT.json').read_text())
assert plan['cache_manifest_sha256']==manifest_sha and plan['candidate_file_count']==480
result=json.loads((root/'reports/VG_METHOD_RESULT.json').read_text())
assert result['status']=='screening_failed_to_establish_benefit' and result['test_scored'] is False
expected={e['file']:e for entries in manifest['splits'].values() for e in entries}
approved={Path(row['path']).name:row for row in plan['candidate_files']}
assert len(expected)==len(approved)==480 and expected.keys()==approved.keys()
assert {p.name for p in cache.iterdir()}==set(expected)|{'manifest.json'}

def signature(s):return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)

def consumers():
 blockers=[];unreadable=[];platform_services=[]
 for p in Path('/proc').iterdir():
  if not p.name.isdigit() or int(p.name)==os.getpid():continue
  try:
   if p.stat().st_uid!=os.getuid():continue
   status=(p/'status').read_text()
   if any(line.startswith('State:') and ('Z ' in line or 'X ' in line) for line in status.splitlines()):continue
   cmd=(p/'cmdline').read_bytes()
   if not cmd:continue
   if b'remote_vg_method.py' in cmd or str(cache).encode() in cmd:
    blockers.append({'pid':int(p.name),'reason':'v1 command/cache argument'})
   fields={k:v.strip() for k,v in (line.split(':',1) for line in status.splitlines() if ':' in line)}
   parent_name=(Path('/proc')/fields['PPid']/'comm').read_text().strip()
   args=[a for a in cmd.split(b'\0') if a];program=Path(args[0].decode(errors='replace')).name
   # Read-only diagnosis established these exact non-dumpable platform roles.
   # Unknown/unreadable application processes still block the deletion.
   daemon=(len(args)==1 and ((fields['Name']=='(sd-pam)' and program=='(sd-pam)' and parent_name=='systemd') or
           (fields['Name']=='sshd' and program.startswith('sshd: ') and parent_name=='sshd') or
           (fields['Name']=='sftp-server' and program=='sftp-server' and parent_name=='sshd')))
   if daemon:
    platform_services.append({'pid':int(p.name),'name':fields['Name'],'parent_name':parent_name});continue
   paths=[]
   try:
    for f in (p/'fd').iterdir():
     try:paths.append(os.readlink(f))
     except FileNotFoundError:pass
   except PermissionError:unreadable.append({'pid':int(p.name),'surface':'fd'})
   try:paths.extend(line.split(maxsplit=5)[-1] for line in (p/'maps').read_text().splitlines() if len(line.split(maxsplit=5))==6)
   except PermissionError:unreadable.append({'pid':int(p.name),'surface':'maps'})
   if any(x==str(cache) or x.startswith(str(cache)+'/') for x in paths):blockers.append({'pid':int(p.name),'reason':'open/mapped cache'})
  except (FileNotFoundError,ProcessLookupError):pass
 return blockers,unreadable,platform_services

blockers,unreadable,platform_services=consumers();assert not blockers,repr(blockers);assert not unreadable,repr(unreadable)
for name,e in expected.items():
 assert Path(name).name==name and name.endswith('.pt') and name.startswith(('train_','val_'))
 p=cache/name;s=p.lstat();a=approved[name]
 assert p.resolve().parent==cache and stat.S_ISREG(s.st_mode) and not p.is_symlink() and s.st_nlink==1
 assert signature(s)==(a['device'],a['inode'],a['bytes'],a['mtime_ns'])
 assert s.st_size==e['bytes'] and a['expected_sha256_from_manifest']==e['sha256']
# Snapshot every other project file's identity/stat before deleting the whitelist.
targets={str(cache/name) for name in expected};protected={}
for folder,dirs,names in os.walk(root,followlinks=False):
 dirs[:]=[d for d in dirs if not (Path(folder)/d).is_symlink()]
 for name in names:
  p=Path(folder)/name
  if str(p) not in targets:protected[p.relative_to(root).as_posix()]=signature(p.lstat())
audit=root/'artifacts/maintenance/vg_v1_cache_retirement_20261022';audit.mkdir(parents=True,exist_ok=False)
(audit/'original_manifest.json').write_bytes(raw)
preservation={'utc':datetime.now(timezone.utc).isoformat(),'authorization':'Human22Oct2026: ok don va tiep tuc; exact prior proposal480V1 cache .pt shards only',
              'manifest_sha256':manifest_sha,'approved_plan':plan,'protected_project_file_signatures':protected,
              'scope':'No checkpoint/log/config/dataset/V2 cache/prediction/test/journal deletion or process signal'}
(audit/'pre_deletion_audit.json').write_text(json.dumps(preservation,indent=2))
free_before=shutil.disk_usage(root).free;deleted=[]
fd=os.open(cache,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
try:
 with (audit/'deletions.jsonl').open('x') as journal:
  for name in sorted(expected):
   a=approved[name];s=os.stat(name,dir_fd=fd,follow_symlinks=False)
   assert stat.S_ISREG(s.st_mode) and s.st_nlink==1 and signature(s)==(a['device'],a['inode'],a['bytes'],a['mtime_ns'])
   os.unlink(name,dir_fd=fd)
   deleted.append(a)
   journal.write(json.dumps({'deleted_path':a['path'],'bytes':a['bytes'],'expected_sha256':a['expected_sha256_from_manifest']})+'\n')
   journal.flush()
  os.fsync(journal.fileno())
finally:os.close(fd)
assert len(deleted)==480 and {p.name for p in cache.iterdir()}=={'manifest.json'}
assert hashlib.sha256(manifest_path.read_bytes()).hexdigest()==manifest_sha
for rel,sig in protected.items():assert signature((root/rel).lstat())==tuple(sig),'protected_file_changed:'+rel
free_after=shutil.disk_usage(root).free
out={'utc':datetime.now(timezone.utc).isoformat(),'status':'COMPLETE','deleted_file_count':len(deleted),
     'deleted_allocated_bytes':sum(a['allocated_bytes'] for a in deleted),'free_before_bytes':free_before,
     'free_after_bytes':free_after,'observed_free_gain_bytes':free_after-free_before,
     'manifest_preserved_sha256':manifest_sha,'other_project_files_unchanged':len(protected),
     'cache_directory_kept':True,'v1_cache_retired':True,'own_active_consumers':blockers,'processes_signalled':False,
     'non_dumpable_platform_services_identified_separately':platform_services,'other_accounts_not_inspected':True,
     'v1_train_or_test_restarted':False,'component_test_rerun':False,'reserve_admission_met':free_after>16*1024**3+128*1024**2,
     'preservation_audit':'artifacts/maintenance/vg_v1_cache_retirement_20261022'}
(audit/'result.json').write_text(json.dumps(out,indent=2));(root/'reports/VG_V1_CACHE_CLEANUP_RESULT.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out))
