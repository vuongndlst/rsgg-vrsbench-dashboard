"""Read-only V2 feature offload proposal, no shard payload or model inference."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,resource,signal,shutil
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});resource.setrlimit(resource.RLIMIT_AS,(512*1024**2,512*1024**2));signal.alarm(45)
root=(Path.home()/'rsg-former').resolve();cache=root/'artifacts/vg_method_v2/features'
assert cache.is_dir() and not cache.is_symlink() and cache.resolve()==cache
raw=(cache/'manifest.json').read_bytes();m=json.loads(raw);rows=[];unknown=[]
for p in cache.iterdir():
 if p.name=='manifest.json':continue
 st=p.lstat()
 if p.is_file() and not p.is_symlink() and p.suffix=='.pt':rows.append({'path':p.relative_to(root).as_posix(),'logical_bytes':st.st_size,'allocated_bytes':st.st_blocks*512,'links':st.st_nlink,'device':st.st_dev,'inode':st.st_ino,'mtime_ns':st.st_mtime_ns})
 else:unknown.append(p.name)
registered=[]
for split,entries in m.get('splits',{}).items():
 for e in entries:
  name=e.get('path',e.get('file',''));assert name and Path(name).name==name
  p=cache/name;st=p.stat();assert st.st_size==e['bytes']
  registered.append({'name':name,'split':split,'bytes':e['bytes'],'expected_sha256':e.get('sha256')})
assert {Path(r['path']).name for r in rows}=={r['name'] for r in registered}
blockers=[];errors=0;known=[]
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
 try:
  if proc.stat().st_uid!=os.getuid():continue
  cmd=(proc/'cmdline').read_bytes();reasons=[]
  for script in ['remote_vg_v2_pipeline.py','remote_vg_v2_speed_pipeline.py','remote_vg_v2_time_extension_v1.py']:
   if script.encode() in cmd:reasons.append('legacy_V2_training_controller')
  targets=[]
  try:
   targets.extend(os.readlink(fd) for fd in (proc/'fd').iterdir())
   targets.extend(x.split(maxsplit=5)[-1] for x in (proc/'maps').read_text().splitlines() if len(x.split(maxsplit=5))==6)
  except OSError:errors+=1
  if any(x==str(cache) or x.startswith(str(cache)+'/') for x in targets):reasons.append('cache_open_or_mapped')
  if reasons:blockers.append({'pid':int(proc.name),'reasons':reasons})
 except OSError:errors+=1
allocated=sum(r['allocated_bytes'] for r in rows);free=shutil.disk_usage(root).free
result={'utc':datetime.now(timezone.utc).isoformat(),'read_only':True,'server_writes':0,'files_deleted':0,'cache_manifest_sha256':hashlib.sha256(raw).hexdigest(),'cache_status':m.get('status'),'candidate_shards':rows,'registered_shards':registered,'candidate_count':len(rows),'candidate_allocated_bytes':allocated,'all_files_have_single_link':all(r['links']==1 for r in rows),'unexpected_entries':unknown,'own_process_blockers':blockers,'own_process_permission_errors':errors,'other_accounts_inspected':False,'disk_free_bytes':free,'projected_free_after_shards_bytes':free+allocated,'requires_byte_exact_local_backup_before_any_server_removal':True,'paused_SGG_seeds_require_restoring_exact_cache_before_resume':True,'current_focus_R1_does_not_reference_feature_cache':True,'manifest_payload_hashes_recomputed':False}
print(json.dumps(result))
