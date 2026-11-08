"""Post-retirement offline verification; no weights/GPU load or generation."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
 OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil,struct,signal
signal.alarm(75);home=Path.home();root=home/'rsg-former';shared=home/'shared-assets'
new=home/'etechs/msvd-tkg-captioning/models/qwen2.5-14b'
repo=shared/'models/huggingface/hub/models--Qwen--Qwen2.5-14B-Instruct'
short=shared/'models/Qwen2.5-14B-Instruct'
assert not os.path.lexists(repo) and not os.path.lexists(short) and new.resolve()==new
runtime=json.loads((root/'configs/qwen_runtime.json').read_text())
assert runtime['model_path']=='${REMOTE_HOME}/etechs/msvd-tkg-captioning/models/qwen2.5-14b'
for name,digest in runtime['auxiliary_files_sha256'].items():
 assert hashlib.sha256((new/name).read_bytes()).hexdigest()==digest,name
index=json.loads((new/'model.safetensors.index.json').read_text());names=sorted(set(index['weight_map'].values()))
assert len(names)==8;shards=[]
for name in names:
 p=new/name
 with p.open('rb') as f:
  header_size=struct.unpack('<Q',f.read(8))[0];assert 0<header_size<16*1024**2
  h=json.loads(f.read(header_size))
 end=max(v['data_offsets'][1] for k,v in h.items() if k!='__metadata__')
 assert p.stat().st_size==8+header_size+end
 shards.append({'name':name,'bytes':p.stat().st_size,'layout_pass':True})
registry_paths=[shared/'ASSET_REGISTRY.json',root/'reports/SHARED_ASSET_REGISTRY.json'];registry_checks=[]
for p in registry_paths:
 data=json.loads(p.read_text());assert data['qwen14b']['path']==data['qwen14b']['load_path']==runtime['model_path']
 registry_checks.append({'path':str(p),'pass':True})
source_checks=[]
for rel in ['scripts/remote_qwen_metadata_lock.py','scripts/remote_qwen_quantization_preflight.py',
 'scripts/remote_caption_runtime_preflight.py','scripts/remote_share_karpathy_assets.py','scripts/remote_shared_qwen_alias.py']:
 text=(root/rel).read_text();compile(text,rel,'exec')
 assert 'etechs/msvd-tkg-captioning/models/qwen2.5-14b' in text
 source_checks.append({'path':rel,'sha256':hashlib.sha256((root/rel).read_bytes()).hexdigest(),'new_path_present':True})
from transformers import AutoConfig,AutoTokenizer
config=AutoConfig.from_pretrained(new,local_files_only=True);tok=AutoTokenizer.from_pretrained(new,local_files_only=True)
tokens=tok.apply_chat_template([{'role':'system','content':'Describe supplied scene graph facts only.'},
 {'role':'user','content':'man holding phone'}],tokenize=True,add_generation_prompt=True,return_dict=False)
assert config.model_type=='qwen2' and len(tokens)==23
print(json.dumps({'status':'PASS_QWEN_NEW_PATH_AFTER_SHARED_RETIREMENT','utc':datetime.now(timezone.utc).isoformat(),
 'new_path':str(new),'old_shared_repository_absent':True,'old_short_alias_absent':True,
 'registry_checks':registry_checks,'source_checks':source_checks,'all8_shard_layouts_pass':True,'shards':shards,
 'tokenizer_load_offline_pass':True,'synthetic_chat_tokens':len(tokens),'auxiliary_sha256_checks_pass':True,
 'full_weight_equality_evidence':'reports/QWEN_RELOCATION_PREFLIGHT_20261025.json; target inodes unchanged at retirement',
 'model_weights_loaded':False,'gpu_used':False,'captions_generated':False,'test_scored':False,
 'disk_free_bytes':shutil.disk_usage(root).free}))
