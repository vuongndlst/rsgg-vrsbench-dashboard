"""Authorized whitelist-only retirement after all981local archive roundtrips passed."""
from pathlib import Path
from datetime import datetime,timezone
import os,json,hashlib,stat,resource,signal,shutil
PLAN=__INLINE_APPROVED_PLAN__
RECEIPT_RAW=__INLINE_VERIFIED_BACKUP_RECEIPT_RAW__
RECEIPT_SHA=__INLINE_VERIFIED_BACKUP_RECEIPT_SHA__
receipt=json.loads(RECEIPT_RAW);assert hashlib.sha256(RECEIPT_RAW.encode()).hexdigest()==RECEIPT_SHA
assert receipt['status']=='PASS_ALL981_LOCAL_ARCHIVES_RESTORED_TO_ORIGINAL_SHA256' and receipt['verified_shard_count']==981
assert receipt['server_deleted_files']==0 and all(r['restored_original_SHA256_pass'] for r in receipt['rows'])
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});resource.setrlimit(resource.RLIMIT_AS,(512*1024**2,512*1024**2));signal.alarm(120)
root=(Path.home()/'rsg-former').resolve();cache=root/'artifacts/vg_method_v2/features';audit=root/'artifacts/maintenance/vg_v2_cache_offload_20261101'
assert cache.resolve()==cache and cache.is_relative_to(root) and cache!=root and not cache.is_symlink()
assert not audit.exists(),'no_automatic_deletion_retry'
raw=(cache/'manifest.json').read_bytes();manifest_sha=hashlib.sha256(raw).hexdigest();assert manifest_sha==PLAN['cache_manifest_sha256']==receipt['cache_manifest_sha256']=='f978799bbd19047b6a74f291aa816f43052251f9b84751f633e8ddf04622abf1'
m=json.loads(raw);expected={e['path']:e for es in m['splits'].values() for e in es};approved={Path(e['path']).name:e for e in PLAN['candidate_shards']};backed={e['name']:e for e in receipt['rows']}
assert len(expected)==len(approved)==len(backed)==981 and expected.keys()==approved.keys()==backed.keys()
assert {p.name for p in cache.iterdir()}==set(expected)|{'manifest.json'}
def sig(s):return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)
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
blockers,unreadable,services=consumers();assert not blockers,repr(blockers);assert not unreadable,repr(unreadable)
for name,e in expected.items():
 assert Path(name).name==name and name.endswith('.pt') and name.startswith(('train_','val_'))
 p=cache/name;s=p.lstat();a=approved[name];b=backed[name]
 assert p.resolve().parent==cache and stat.S_ISREG(s.st_mode) and s.st_uid==os.getuid() and s.st_nlink==1 and not p.is_symlink()
 assert sig(s)==(a['device'],a['inode'],a['logical_bytes'],a['mtime_ns'])
 assert s.st_size==e['bytes']==b['raw_bytes'] and e['sha256']==b['original_sha256'] and b['restored_original_SHA256_pass']
targets={str(cache/n) for n in expected};protected={}
for folder,dirs,names in os.walk(root,followlinks=False):
 dirs[:]=[d for d in dirs if not (Path(folder)/d).is_symlink()]
 for name in names:
  p=Path(folder)/name
  if str(p) not in targets:protected[p.relative_to(root).as_posix()]=sig(p.lstat())
# Cleanup-specific audit is required before unlink and is the only new server write below the restored reserve.
assert shutil.disk_usage(root).free>32*1024**2,'cleanup_audit_headroom'
audit.mkdir(parents=True,exist_ok=False);(audit/'original_manifest.json').write_bytes(raw);(audit/'verified_local_backup_receipt.json').write_text(RECEIPT_RAW)
preservation={'utc':datetime.now(timezone.utc).isoformat(),'authorization':'Human1Nov2026: save V2 cache locally then continue server work, approving prior981-shard verified backup/removal proposal','target_cache_relative':'artifacts/vg_method_v2/features','target_names':sorted(expected),'local_receipt_sha256':RECEIPT_SHA,'protected_file_signatures':protected,'scope':'Only981verified .pt shards; no manifest/checkpoint/source/config/log/prediction/journal/other-asset deletion, no process signal','non_dumpable_platform_roles_identified':services}
(audit/'before_deletion.json').write_text(json.dumps(preservation,indent=2));free_before=shutil.disk_usage(root).free;deleted=[]
fd=os.open(cache,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
try:
 with (audit/'deletions.jsonl').open('x') as journal:
  for name in sorted(expected):
   a=approved[name];s=os.stat(name,dir_fd=fd,follow_symlinks=False)
   assert stat.S_ISREG(s.st_mode) and s.st_nlink==1 and sig(s)==(a['device'],a['inode'],a['logical_bytes'],a['mtime_ns'])
   os.unlink(name,dir_fd=fd);deleted.append(name);journal.write(json.dumps({'deleted_path':a['path'],'bytes':a['logical_bytes'],'raw_sha256':expected[name]['sha256'],'local_archive_file':backed[name]['archive_file'],'local_archive_sha256':backed[name]['archive_sha256']})+'\n');journal.flush()
  os.fsync(journal.fileno())
finally:os.close(fd)
assert len(deleted)==981 and {p.name for p in cache.iterdir()}=={'manifest.json'}
assert hashlib.sha256((cache/'manifest.json').read_bytes()).hexdigest()==manifest_sha
for rel,s in protected.items():assert sig((root/rel).lstat())==tuple(s),'protected_file_changed:'+rel
free_after=shutil.disk_usage(root).free
result={'status':'COMPLETE_981_VERIFIED_V2_SHARDS_OFFLOADED_AND_SERVER_COPIES_REMOVED','utc':datetime.now(timezone.utc).isoformat(),'deleted_files':981,'deleted_allocated_bytes':sum(e['allocated_bytes'] for e in approved.values()),'free_before_bytes':free_before,'free_after_bytes':free_after,'observed_free_gain_bytes':free_after-free_before,'manifest_preserved_sha256':manifest_sha,'local_backup_receipt_sha256':RECEIPT_SHA,'local_archives_verified':981,'other_project_file_signatures_unchanged':len(protected),'cache_directory_kept':True,'V2_seeds_require_exact_cache_restore':True,'SGDet_or_caption_TEST_rerun':False,'processes_signalled':False,'own_active_consumers':blockers,'non_dumpable_platform_roles_identified':services,'other_accounts_not_inspected':True,'reserve_restored':free_after>=17*1024**3,'audit':'artifacts/maintenance/vg_v2_cache_offload_20261101'}
(audit/'result.json').write_text(json.dumps(result,indent=2));(root/'reports/VG_V2_CACHE_OFFLOAD_RESULT_20261101.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
