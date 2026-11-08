"""Validate existing Qwen shard headers and tokenizer metadata, without loading the model."""
from pathlib import Path
from datetime import datetime,timezone
import json,struct,hashlib,re

home=Path.home();revision='cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8'
folder=home/'etechs/msvd-tkg-captioning/models/qwen2.5-14b'
def alias(p):return re.sub(r'/(home|users)/[^/\s]+',r'/\1/${ACCOUNT}',str(p).replace(str(home),'${REMOTE_HOME}').replace(home.name,'${ACCOUNT}'))
raw=(folder/'model.safetensors.index.json').read_bytes();index=json.loads(raw);names=sorted(set(index['weight_map'].values()));shards=[]
for name in names:
    path=folder/name
    with path.open('rb') as f:
        header_size=struct.unpack('<Q',f.read(8))[0]
        if not 0<header_size<16*1024**2:raise ValueError('unexpected_safetensors_header_size')
        header_bytes=f.read(header_size);header=json.loads(header_bytes)
    tensors={k:v for k,v in header.items() if k!='__metadata__'};end=max(v['data_offsets'][1] for v in tensors.values())
    expected=8+header_size+end;actual=path.stat().st_size
    shards.append({'name':name,'file_bytes':actual,'expected_bytes_from_header':expected,'layout_complete':actual==expected,'header_sha256':hashlib.sha256(header_bytes).hexdigest(),'tensor_count':len(tensors),'dtypes':sorted(set(v['dtype'] for v in tensors.values()))})
tokenizer={}
for name in ['config.json','tokenizer_config.json','tokenizer.json','vocab.json','merges.txt','generation_config.json']:
    path=folder/name;tokenizer[name]={'present':path.is_file(),'bytes':path.stat().st_size if path.is_file() else None}
print(json.dumps({'audited_at_utc':datetime.now(timezone.utc).isoformat(),'status':'METADATA_PASS' if all(x['layout_complete'] for x in shards) and tokenizer['tokenizer_config.json']['present'] and tokenizer['tokenizer.json']['present'] else 'INCOMPLETE','model':'Qwen/Qwen2.5-14B-Instruct','model_kind':'text_only_graph_to_text','revision':revision,'path_alias':alias(folder),'index_sha256':hashlib.sha256(raw).hexdigest(),'shards':shards,'total_shard_bytes':sum(x['file_bytes'] for x in shards),'tokenizer_files':tokenizer,'full_weight_checksums_computed':False,'model_loaded':False,'gpu_inference_verified':False,'policy':'reuse this existing snapshot without copying/downloading; runtime load probe after VG GPU work completes'}))
