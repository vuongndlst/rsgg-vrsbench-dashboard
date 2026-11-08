"""Read-only C-backed model discovery over accessible real filesystems."""
from pathlib import Path
from datetime import datetime,timezone
import os,time,json,re,stat,subprocess,hashlib

home=Path.home();start=time.monotonic()
def alias(p):return re.sub(r'/(home|users)/[^/\s]+',r'/\1/${ACCOUNT}',str(p).replace(str(home),'${REMOTE_HOME}').replace(home.name,'${ACCOUNT}'))
excluded=['/proc','/sys','/dev','/run'];pruned=['.git','node_modules','__pycache__','site-packages','dist-packages']
args=['find','/','(']
for i,path in enumerate(excluded):
    if i:args+=['-o']
    args+=['-path',path]
for name in pruned:args+=['-o','-name',name]
args+=[')','-prune','-o','(','-type','f','-o','-type','l',')','(']
patterns=['*.safetensors','*.gguf','*.ggml','*.ckpt','*.pth','*.pt','pytorch_model*.bin','model*.bin','config.json','*.index.json','*.zip','*.tar','*.tar.gz','*.tar.zst']
for i,name in enumerate(patterns):
    if i:args+=['-o']
    args+=['-name',name]
args+=['-o','-path','*/huggingface/*/blobs/*','-o','-path','*/.ollama/models/blobs/*',')','-print0']
p=subprocess.Popen(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
timed_out=False
try:out,err=p.communicate(timeout=150)
except subprocess.TimeoutExpired:
    timed_out=True;p.kill();out,err=p.communicate()
paths=[Path(os.fsdecode(x)) for x in out.split(b'\0') if x];weights=[];configs=[];large=[];gguf=[]
for path in paths:
    try:
        st=path.stat()
        if not stat.S_ISREG(st.st_mode):continue
    except OSError:continue
    name=path.name
    weight=name.endswith(('.safetensors','.gguf','.ggml','.ckpt','.pth','.pt')) or bool(re.match(r'(pytorch_model|model)(?:-\d+-of-\d+)?\.bin$',name))
    record={'path_alias':alias(path),'bytes':st.st_size,'allocated_bytes':getattr(st,'st_blocks',0)*512,'inode_key':f'{st.st_dev}:{st.st_ino}','symlink':path.is_symlink(),'target_alias':alias(path.resolve()) if path.is_symlink() else None}
    if weight and st.st_size>=32*1024**2:
        record['kind']='feature_cache' if 'vg_method_v1/features' in str(path) else 'model_artifact';weights.append(record)
    if st.st_size>=2*1024**3 and not weight:large.append(record)
    if '.ollama/models/blobs/' in str(path) and st.st_size>=32*1024**2:
        try:
            with path.open('rb') as f:magic=f.read(4)
            if magic==b'GGUF':gguf.append(record)
        except OSError:pass
    if (name=='config.json' or name.endswith('.index.json')) and st.st_size<2000000:
        try:
            raw=path.read_bytes();data=json.loads(raw)
            if name=='config.json' and ('model_type' in data or 'architectures' in data):
                info={k:data[k] for k in ['architectures','model_type','torch_dtype','dtype','hidden_size','num_hidden_layers','quantization_config'] if k in data};info['vision_config_present']='vision_config' in data
                configs.append({'path_alias':alias(path.parent),'config':info,'config_sha256':hashlib.sha256(raw).hexdigest()})
            elif 'weight_map' in data:
                names=sorted(set(data['weight_map'].values()));configs.append({'path_alias':alias(path.parent),'index_name':name,'index_sha256':hashlib.sha256(raw).hexdigest(),'declared_total_bytes':data.get('metadata',{}).get('total_size'),'shard_count':len(names),'shards_present':sum((path.parent/n).is_file() for n in names)})
        except (OSError,ValueError):pass
groups={}
for w in weights:
    parent=w['path_alias'].rsplit('/',1)[0];g=groups.setdefault(parent,{'files':0,'logical_bytes':0,'unique_inodes':{},'kind':w['kind']});g['files']+=1;g['logical_bytes']+=w['bytes'];g['unique_inodes'][w['inode_key']]=w['allocated_bytes']
for g in groups.values():g['allocated_unique_bytes']=sum(g.pop('unique_inodes').values())
unique={w['inode_key']:w for w in weights}
disk=subprocess.run(['df','-B1','--output=fstype,size,used,avail,target'],capture_output=True,text=True,timeout=10).stdout
print(json.dumps({'audited_at_utc':datetime.now(timezone.utc).isoformat(),'mode':'read_only_no_weight_load_no_deletion','scan':{'root':'/','excluded_virtual_roots':excluded,'excluded_directory_names':pruned,'min_weight_file_bytes':32*1024**2,'follow_directory_symlinks':False,'completed_search':not timed_out,'find_returncode':p.returncode,'wall_seconds':time.monotonic()-start,'time_limit_seconds':150,'candidate_files':len(paths),'permission_error_samples':alias(err.decode(errors='replace')).splitlines()[:25],'completeness':'accessible real filesystem entries; permission-denied and pruned dependencies excluded; directory symlinks not traversed'},'filesystem_snapshot':alias(disk),'model_configs':configs,'weights':weights,'weight_directory_totals':groups,'unique_weight_allocated_bytes':sum(w['allocated_bytes'] for w in unique.values()),'large_other_files':large,'ollama_gguf_blobs':gguf},ensure_ascii=False))
