"""Read-only inventory of accessible real filesystems, without loading or hashing weights."""
from pathlib import Path
from datetime import datetime,timezone
import os,time,json,re,stat,subprocess,hashlib

home=str(Path.home());start=time.monotonic();limit=150;errors=[];weights=[];configs=[];large=[];directories=0;files=0;finished=True
def alias(p):return re.sub(r'/(home|users)/[^/\s]+',r'/\1/${ACCOUNT}',str(p).replace(home,'${REMOTE_HOME}').replace(Path(home).name,'${ACCOUNT}'))
def err(e):
    if len(errors)<30:errors.append({'path_alias':alias(e.filename or ''),'error_type':type(e).__name__})
excluded={'/proc','/sys','/dev','/run'}
excluded_names={'.git','node_modules','__pycache__','site-packages','dist-packages'}
for root,dirs,names in os.walk('/',topdown=True,onerror=err,followlinks=False):
    directories+=1
    dirs[:]=[d for d in dirs if os.path.join(root,d) not in excluded and d not in excluded_names]
    if time.monotonic()-start>limit:finished=False;break
    for name in names:
        files+=1;p=Path(root)/name
        isweight=name.endswith(('.safetensors','.gguf','.ggml','.ckpt','.pth','.pt')) or bool(re.match(r'(pytorch_model|model)(?:-\d+-of-\d+)?\.bin$',name))
        isconfig=name=='config.json' or name.endswith('.safetensors.index.json') or name=='pytorch_model.bin.index.json'
        try:
            st=p.stat()
            if not stat.S_ISREG(st.st_mode):continue
        except OSError:continue
        if isweight and st.st_size>=32*1024**2:
            weights.append({'path_alias':alias(p),'bytes':st.st_size,'allocated_bytes':getattr(st,'st_blocks',0)*512,'inode_key':f'{st.st_dev}:{st.st_ino}','symlink':p.is_symlink(),'target_alias':alias(p.resolve()) if p.is_symlink() else None})
        if st.st_size>=2*1024**3 and not isweight:large.append({'path_alias':alias(p),'bytes':st.st_size,'allocated_bytes':getattr(st,'st_blocks',0)*512,'inode_key':f'{st.st_dev}:{st.st_ino}'})
        if isconfig and st.st_size<2000000:
            try:
                raw=p.read_bytes();data=json.loads(raw)
                if name=='config.json' and ('model_type' in data or 'architectures' in data):
                    info={k:data[k] for k in ['architectures','model_type','torch_dtype','dtype','hidden_size','num_hidden_layers','quantization_config'] if k in data}
                    info['vision_config_present']='vision_config' in data
                    configs.append({'path_alias':alias(p.parent),'config':info,'config_sha256':hashlib.sha256(raw).hexdigest()})
                elif 'weight_map' in data:
                    shards=sorted(set(data['weight_map'].values()))
                    configs.append({'path_alias':alias(p.parent),'index_name':name,'index_sha256':hashlib.sha256(raw).hexdigest(),'declared_total_bytes':data.get('metadata',{}).get('total_size'),'shard_count':len(shards),'shards_present':sum((p.parent/n).is_file() for n in shards)})
            except (OSError,ValueError):pass
    if files>=1500000:finished=False;break
try:
    disk=subprocess.run(['df','-B1','--output=fstype,size,used,avail,target'],capture_output=True,text=True,timeout=10).stdout
except (OSError,subprocess.TimeoutExpired):disk='unavailable'
unique={w['inode_key']:w for w in weights}
groups={}
for w in weights:
    parent=w['path_alias'].rsplit('/',1)[0];g=groups.setdefault(parent,{'files':0,'logical_bytes':0,'unique_inodes':{}})
    g['files']+=1;g['logical_bytes']+=w['bytes'];g['unique_inodes'][w['inode_key']]=w['allocated_bytes']
for g in groups.values():g['allocated_unique_bytes']=sum(g.pop('unique_inodes').values())
print(json.dumps({'audited_at_utc':datetime.now(timezone.utc).isoformat(),'mode':'read_only_no_weight_load_no_deletion','scan':{'root':'/','excluded_virtual_roots':sorted(excluded),'excluded_directory_names':sorted(excluded_names),'min_weight_file_bytes':32*1024**2,'follow_directory_symlinks':False,'completed_walk':finished,'wall_seconds':time.monotonic()-start,'time_limit_seconds':limit,'directories':directories,'files':files,'permission_error_samples':errors,'completeness':'accessible entries only; dependency packages, permission-denied and directory symlink targets outside traversed roots are not guaranteed'},'filesystem_snapshot':alias(disk),'model_configs':configs,'weights':weights,'weight_directory_totals':groups,'unique_weight_allocated_bytes':sum(w['allocated_bytes'] for w in unique.values()),'large_nonweight_files':large},ensure_ascii=False))
