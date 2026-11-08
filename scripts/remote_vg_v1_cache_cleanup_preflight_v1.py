"""Read-only exact deletion proposal for retired V1 feature shards only."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib,json,os,resource,shutil,signal
resource.setrlimit(resource.RLIMIT_AS,(256*1024**2,256*1024**2))
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});signal.alarm(60)
root=(Path.home()/'rsg-former').resolve();base=root/'artifacts/vg_method_v1';cache=base/'features'
assert not cache.is_symlink() and cache.resolve()==cache and cache.is_dir()
assert cache.is_relative_to(root) and cache!=root
manifest_path=cache/'manifest.json';raw=manifest_path.read_bytes();manifest=json.loads(raw)
assert manifest['status']=='complete'
rows=[]
for split,entries in manifest['splits'].items():
 for e in entries:
  name=e['file'];assert Path(name).name==name and name.endswith('.pt') and name.startswith(split+'_')
  p=cache/name;s=p.lstat();assert p.is_file() and not p.is_symlink() and p.resolve().parent==cache
  assert s.st_size==e['bytes'] and s.st_nlink==1
  rows.append({'path':p.relative_to(root).as_posix(),'split':split,'bytes':s.st_size,'allocated_bytes':s.st_blocks*512,'device':s.st_dev,'inode':s.st_ino,'mtime_ns':s.st_mtime_ns,'expected_sha256_from_manifest':e['sha256']})
assert len({r['path'] for r in rows})==len(rows)
unknown=[p.name for p in cache.iterdir() if p.name!='manifest.json' and p.name not in {Path(r['path']).name for r in rows}]
assert not unknown,'unexpected_cache_entries'
result_path=root/'reports/VG_METHOD_RESULT.json';result=json.loads(result_path.read_text())
blockers=[];unreadable_processes=0
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
 try:
  if proc.stat().st_uid!=os.getuid():continue
  cmd=(proc/'cmdline').read_bytes();reasons=[]
  if b'remote_vg_method.py' in cmd or str(cache).encode() in cmd:reasons.append('v1_command_or_cache_argument')
  paths=[]
  try:
   paths.extend(os.readlink(p) for p in (proc/'fd').iterdir())
   paths.extend(line.split(maxsplit=5)[-1] for line in (proc/'maps').read_text().splitlines() if len(line.split(maxsplit=5))==6)
  except OSError:unreadable_processes+=1
  if any(p==str(cache) or p.startswith(str(cache)+'/') for p in paths):reasons.append('open_or_mapped_v1_cache')
  if reasons:blockers.append({'pid':int(proc.name),'reasons':reasons})
 except OSError:unreadable_processes+=1
allocated=sum(r['allocated_bytes'] for r in rows);free=shutil.disk_usage(root).free
out={'utc':datetime.now(timezone.utc).isoformat(),'read_only':True,'deletion_performed':False,
     'proposal':'Remove only manifest-listed retired V1 feature .pt shards; preserve manifest, V1 source/config/logs/metrics/checkpoints and all V2/component assets',
     'v1_terminal_result_status':result.get('status'),'v1_test_scored':result.get('test_scored'),
     'cache_manifest_sha256':hashlib.sha256(raw).hexdigest(),'cache_manifest_preserved':True,
     'candidate_files':rows,'candidate_file_count':len(rows),'candidate_allocated_bytes':allocated,
     'split_image_counts':{k:sum(len(e['image_ids']) for e in entries) for k,entries in manifest['splits'].items()},
     'free_bytes':free,'estimated_free_after_bytes':free+allocated,'own_process_blockers':blockers,
     'unreadable_own_process_checks':unreadable_processes,'other_accounts_not_inspected':True,
     'expected_sha_source':'Existing manifest, shard content hashes not recomputed in this metadata-only preflight',
     'consequence':'V1 cached features unavailable after deletion; reproducible regeneration would cost compute and is not authorized by this proposal. No V1 train/test restart.'}
print(json.dumps(out))
