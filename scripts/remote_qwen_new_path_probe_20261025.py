"""Offline new-directory config/tokenizer probe plus bounded source-path search."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
 OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,signal,time
signal.alarm(100);start=time.monotonic();home=Path.home()
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
model=home/'etechs/msvd-tkg-captioning/models/qwen2.5-14b'
assert model.resolve()==model and not model.is_symlink()
assert hashlib.sha256((model/'config.json').read_bytes()).hexdigest()=='0f2085dbbe2ee251bd6a6a0797d84a6ce34436044d629aa3cba793b43d311a9e'
assert hashlib.sha256((model/'model.safetensors.index.json').read_bytes()).hexdigest()=='46627f7a9e851d9cede6a7a4c49999082148ced9c22d2b9cfaaa5ba8b65dc68f'
from transformers import AutoConfig,AutoTokenizer
config=AutoConfig.from_pretrained(model,local_files_only=True)
tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True)
tokens=tokenizer.apply_chat_template([{'role':'system','content':'Describe supplied scene graph facts only.'},
 {'role':'user','content':'man holding phone'}],tokenize=True,add_generation_prompt=True,return_dict=False)
assert config.model_type=='qwen2' and len(tokens)>0
skip={'.git','.venv','venv','node_modules','site-packages','dist-packages','.tools','reports','models',
 'datasets','data','outputs','artifacts','.cache','shared-assets','.codex','.aws','.ssh','.local','.config','__pycache__'}
matches=[];visited=0;complete=True;errors=0
for folder,dirs,names in os.walk(home,followlinks=False):
 dirs[:]=[d for d in dirs if d not in skip and not (Path(folder)/d).is_symlink()]
 if time.monotonic()-start>80:complete=False;break
 for name in names:
  p=Path(folder)/name
  if p.suffix.lower() not in {'.py','.json','.yaml','.yml','.toml','.sh'}:continue
  if any(s in name.lower() for s in ['credential','secret','tokenizer','vocab','merges','token.']):continue
  try:
   if p.is_symlink() or p.stat().st_size>2*1024**2:continue
   visited+=1;text=p.read_text(encoding='utf-8',errors='replace')
   if 'shared-assets' in text and ('Qwen2.5-14B' in text or 'models--Qwen' in text):
    matches.append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
      'line_numbers':[i for i,line in enumerate(text.splitlines(),1) if 'Qwen2.5-14B' in line or 'models--Qwen' in line]})
  except OSError:errors+=1
print(json.dumps({'status':'NEW_QWEN_PATH_OFFLINE_CONFIG_TOKENIZER_PASS','utc':datetime.now(timezone.utc).isoformat(),
 'model_path':str(model),'model_type':config.model_type,'tokenizer_class':type(tokenizer).__name__,
 'synthetic_chat_tokens':len(tokens),'weights_loaded':False,'gpu_used':False,'test_scored':False,
 'source_scan':{'complete':complete,'visited_text_files':visited,'errors':errors,'matches':matches,
 'scope':'home code/config excluding model/data/artifact/report/dependency/private settings trees; no credential files'},
 'wall_seconds':time.monotonic()-start}))
