"""Bounded, read-only discovery of existing Qwen model metadata; never load weights."""
from pathlib import Path
from datetime import datetime,timezone
from collections import deque
import json,re,subprocess,importlib.metadata,hashlib

home=Path.home()
roots=[home/'.cache/huggingface/hub',home/'etechs',home/'fair-project',home/'models',home/'sbir',Path('/data'),Path('/mnt')]
queue=deque((p,0) for p in roots);seen=set();models=[];budget=180
def alias(p):
    return re.sub(r'/(home|users)/[^/\s]+',r'/\1/${ACCOUNT}',str(p).replace(str(home),'${REMOTE_HOME}'))
while queue and len(seen)<budget:
    folder,depth=queue.popleft()
    if str(folder) in seen:continue
    seen.add(str(folder))
    try:entries=list(folder.iterdir())[:250]
    except OSError:continue
    qwen='qwen' in str(folder).lower()
    if qwen:
        config=folder/'config.json'
        try:
            if config.is_file() and config.stat().st_size<=65536:
                data=json.loads(config.read_text())
                keys=['architectures','model_type','torch_dtype','dtype','hidden_size','num_hidden_layers','num_attention_heads','vocab_size','quantization_config']
                info={k:data[k] for k in keys if k in data}
                info['vision_config_present']='vision_config' in data
                weights=[p for p in entries if p.is_file() and p.suffix in ['.safetensors','.bin','.gguf']]
                indices=[]
                for index in entries:
                    if index.name.endswith('.index.json') and index.is_file() and index.stat().st_size<2000000:
                        raw=index.read_bytes();idx=json.loads(raw);names=sorted(set(idx.get('weight_map',{}).values()))
                        indices.append({'name':index.name,'sha256':hashlib.sha256(raw).hexdigest(),'declared_total_bytes':idx.get('metadata',{}).get('total_size'),'shards':len(names),'shards_present':sum((folder/n).is_file() for n in names)})
                weight_entries=[p for p in entries if p.suffix in ['.safetensors','.bin','.gguf']]
                models.append({'path_alias':alias(folder),'config':info,'weight_files':len(weights),'weight_bytes_in_directory':sum(p.stat().st_size for p in weights),'weight_entries':len(weight_entries),'broken_weight_symlinks':sum(p.is_symlink() and not p.exists() for p in weight_entries),'weight_indices':indices,'tokenizer_config_present':(folder/'tokenizer_config.json').is_file()})
        except (OSError,ValueError):pass
    if depth<6:
        for p in entries:
            try:isdir=p.is_dir()
            except OSError:continue
            if not isdir or p.name in ['.git','.ssh','.aws','.venv','node_modules','site-packages','__pycache__','blobs']:continue
            if depth==0 or qwen or re.search(r'qwen|models|weights|checkpoint|huggingface|snapshots|cache|caption|research|project|storage',p.name,re.I):queue.append((p,depth+1))
versions={}
for name in ['torch','transformers','accelerate','bitsandbytes','pycocoevalcap','pycocotools']:
    try:versions[name]=importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:versions[name]=None
try:
    gpu=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10).stdout.strip()
except (OSError,subprocess.TimeoutExpired):gpu='unavailable'
print(json.dumps({'audited_at_utc':datetime.now(timezone.utc).isoformat(),'mode':'read_only_metadata_no_model_load','qwen_candidates':models,'search':{'directories_visited':len(seen),'budget':budget,'pending_directories':len(queue),'depth_limit':6,'exhaustive':False},'project_environment_versions':versions,'gpu_snapshot':gpu}))
