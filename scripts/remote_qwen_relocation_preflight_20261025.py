"""Read-only full-file equality and consumer checks before authorized Qwen retirement."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,resource,signal,subprocess,time
start=time.monotonic();home=Path.home().resolve()
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});os.nice(10)
resource.setrlimit(resource.RLIMIT_AS,(384*1024**2,384*1024**2));signal.alarm(260)
repo=home/'shared-assets/models/huggingface/hub/models--Qwen--Qwen2.5-14B-Instruct'
old=repo/'snapshots/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8'
new=home/'etechs/msvd-tkg-captioning/models/qwen2.5-14b'
assert repo.resolve()==repo and new.resolve()==new and new.is_dir()
assert not (new==repo or repo in new.parents or new in repo.parents)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
 return h.hexdigest()
oldidx=json.loads((old/'model.safetensors.index.json').read_text())
newidx=json.loads((new/'model.safetensors.index.json').read_text())
assert oldidx==newidx
names=sorted(set(oldidx['weight_map'].values()));assert len(names)==8
aux=['model.safetensors.index.json','config.json','tokenizer_config.json','tokenizer.json','vocab.json','merges.txt','generation_config.json']
results=[]
for name in aux+names:
 assert Path(name).name==name
 a,b=old/name,new/name
 assert a.is_file() and b.is_file(),name
 sa,sb=a.stat(),b.stat()
 assert sa.st_size==sb.st_size,name
 ha,hb=sha(a),sha(b)
 assert ha==hb,'file_content_mismatch:'+name
 results.append({'name':name,'bytes':sa.st_size,'old_sha256':ha,'new_sha256':hb,'equal':True,
  'old_inode':[sa.st_dev,sa.st_ino],'new_inode':[sb.st_dev,sb.st_ino]})
users=[];unreadable=0
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
 try:
  if proc.stat().st_uid!=os.getuid():continue
  cwd=os.readlink(proc/'cwd');cmd=(proc/'cmdline').read_bytes();maps=(proc/'maps').read_bytes()
  reason=[]
  if str(repo).encode() in cmd or b'shared-assets/models/Qwen2.5-14B-Instruct' in cmd:reason.append('command')
  if str(repo).encode() in maps:reason.append('mapped_weights')
  if cwd==str(repo) or cwd.startswith(str(repo)+'/'):reason.append('cwd')
  for fd in (proc/'fd').iterdir():
   try:
    target=os.readlink(fd)
    if target.startswith(str(repo)+'/'):reason.append('open_file');break
   except OSError:pass
  if reason:users.append({'pid':int(proc.name),'reasons':sorted(set(reason))})
 except OSError:unreadable+=1
# Read only matching code/config paths, not contents or credential/environment files.
needles=['shared-assets/models/Qwen2.5-14B-Instruct','shared-assets/models/huggingface/hub/models--Qwen--Qwen2.5-14B-Instruct']
args=['rg','-l','--hidden','--max-filesize','2M','-g','*.py','-g','*.json','-g','*.yaml','-g','*.yml','-g','*.toml','-g','*.sh']
for excluded in ['.git','.venv','venv','node_modules','site-packages','.tools','reports','research/references','models','datasets','data','outputs','artifacts','.cache','shared-assets']:
 args+=['-g','!**/'+excluded+'/**']
for needle in needles:args+=['-e',needle]
args.append(str(home))
try:
 r=subprocess.run(args,capture_output=True,text=True,timeout=40)
 matches=[x for x in r.stdout.splitlines() if x]
 scan={'returncode':r.returncode,'matching_source_config_paths':matches,'scope':'home source/config excluding data/model/report/artifact/dependency trees; exact slash paths only'}
except (FileNotFoundError,subprocess.TimeoutExpired) as e:scan={'error_type':type(e).__name__,'matching_source_config_paths':[]}
print(json.dumps({'status':'PREFLIGHT_EQUALITY_PASS' if not users else 'PREFLIGHT_ACTIVE_CONSUMER_BLOCKED',
 'utc':datetime.now(timezone.utc).isoformat(),'old_repository':str(repo),'new_directory':str(new),
 'files':results,'all_8_shards_and_7_auxiliary_files_byte_identical':True,'own_processes_using_old_repository':users,
 'unreadable_own_processes':unreadable,'source_reference_scan':scan,'model_weights_loaded':False,
 'files_deleted':0,'server_files_written':0,'wall_seconds':time.monotonic()-start}))
