"""Explicitly authorized bounded Qwen BF16 inference pilot; synthetic inputs only."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,resource,shutil,subprocess,sys,time
ROOT=Path.home()/'rsg-former';ART=ROOT/'artifacts/vg_caption_qwen_gpu_pilot_v2_20261025'
assert not ART.exists(),'pilot_already_exists_do_not_duplicate'
assert shutil.disk_usage(ROOT).free>=17*1024**3
free=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
assert int(free.splitlines()[0])>=38*1024,'free_gpu_admission_38GiB'
ART.mkdir();lock={'utc':datetime.now(timezone.utc).isoformat(),'authorization':'Human25Oct explicitly approved GPU36GiB/allocator32GiB/RSS12GiB/CPU4 when resources available',
 'synthetic_inputs_only':True,'benchmark_test_access':False,'maximum_wall_seconds':240,'loader_repair':'stream one safetensor to CUDA at a time; original BF16 unchanged; V1 RSS-cap stop preserved','maximum_new_tokens':32,
 'allocator_gib':32,'own_gpu_gib':36,'rss_gib':12,'cpu':4,'model_path':'${REMOTE_HOME}/etechs/msvd-tkg-captioning/models/qwen2.5-14b'}
(ART/'preregistration.json').write_text(json.dumps(lock,indent=2))
child=r'''
import os
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',CUDA_CACHE_DISABLE='1')
from pathlib import Path
import json,time,resource,sys
os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:4]))
import torch
torch.set_num_threads(4);torch.set_num_interop_threads(1)
torch.cuda.set_per_process_memory_fraction(32*1024**3/torch.cuda.get_device_properties(0).total_memory)
from transformers import AutoModelForCausalLM,AutoTokenizer
p=Path.home()/'etechs/msvd-tkg-captioning/models/qwen2.5-14b';start=time.monotonic()
t=AutoTokenizer.from_pretrained(p,local_files_only=True)
from transformers import AutoConfig,GenerationConfig
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from safetensors import safe_open
import gc
cfg=AutoConfig.from_pretrained(p,local_files_only=True);cfg._attn_implementation='sdpa'
with init_empty_weights():m=AutoModelForCausalLM.from_config(cfg,dtype=torch.bfloat16)
expected=dict(m.named_parameters());index=json.loads((p/'model.safetensors.index.json').read_text());loaded_names=set()
assert set(index['weight_map'])==set(expected),'model_parameter_keys_mismatch'
for filename in sorted(set(index['weight_map'].values())):
 with safe_open(p/filename,framework='pt',device='cpu') as stream:
  for name in stream.keys():
   tensor=stream.get_tensor(name)
   assert tuple(tensor.shape)==tuple(expected[name].shape) and tensor.dtype==torch.bfloat16
   set_module_tensor_to_device(m,name,'cuda:0',value=tensor,dtype=torch.bfloat16)
   loaded_names.add(name);del tensor;gc.collect()
 gc.collect()
assert loaded_names==set(expected) and not any(t.is_meta for t in m.parameters())
m.to('cuda:0').eval();m.generation_config=GenerationConfig.from_pretrained(p,local_files_only=True)

loaded=time.monotonic()-start;rows=[]
for facts in ['man holding phone','shirt[color=red]; man wearing shirt','cat[color=black,white] on chair']:
 messages=[{'role':'system','content':'Use only supplied facts. Return one short English region caption, no explanation.'},{'role':'user','content':'FACTS: '+facts+'\nCaption:'}]
 inputs=t.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_tensors='pt',return_dict=True).to('cuda')
 before=time.monotonic()
 with torch.inference_mode():out=m.generate(**inputs,do_sample=False,num_beams=1,max_new_tokens=32,pad_token_id=t.eos_token_id,use_cache=True)
 new=out[0,inputs['input_ids'].shape[-1]:];rows.append({'facts':facts,'caption':t.decode(new,skip_special_tokens=True),'new_tokens':len(new),'seconds':time.monotonic()-before})
 assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=12*1024**3,'rss_cap'
print(json.dumps({'status':'QWEN_BF16_SYNTHETIC_GPU_PILOT_PASS','load_seconds':loaded,'rows':rows,'peak_allocator_bytes':torch.cuda.max_memory_allocated(),
 'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'wall_seconds':time.monotonic()-start,'test_scored':False}))
'''
env=dict(os.environ);env.update(PYTHONDONTWRITEBYTECODE='1',CUDA_CACHE_DISABLE='1')
start=time.monotonic();reason=None;peak_gpu=0;peak_rss=0
with (ART/'stdout.log').open('w') as stdout,(ART/'stderr.log').open('w') as stderr:
 p=subprocess.Popen([sys.executable,'-B','-c',child],stdout=stdout,stderr=stderr,env=env,start_new_session=True)
 (ART/'child.pid').write_text(str(p.pid))
 while p.poll() is None:
  try:
   status=(Path('/proc')/str(p.pid)/'status').read_text()
   rss=int(next(line for line in status.splitlines() if line.startswith('VmRSS:')).split()[1])*1024;peak_rss=max(peak_rss,rss)
   q=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5)
   gpu=max([int(x.split(',')[1])*1024**2 for x in q.stdout.splitlines() if int(x.split(',')[0])==p.pid] or [0]);peak_gpu=max(peak_gpu,gpu)
   if rss>12*1024**3:reason='rss12GiB_cap'
   elif gpu>36*1024**3:reason='gpu36GiB_cap'
  except (OSError,StopIteration):pass
  if time.monotonic()-start>240:reason='pilot_wall_cap'
  if reason:p.terminate();break
  time.sleep(1)
 try:p.wait(timeout=10)
 except subprocess.TimeoutExpired:p.kill();p.wait()
lines=(ART/'stdout.log').read_text().splitlines();result=None
for line in reversed(lines):
 try:result=json.loads(line);break
 except ValueError:pass
report={'status':result['status'] if p.returncode==0 and result else 'QWEN_GPU_PILOT_FAILED','utc':datetime.now(timezone.utc).isoformat(),
 'returncode':p.returncode,'guard_reason':reason,'peak_nvml_bytes':peak_gpu,'peak_watchdog_rss_bytes':peak_rss,
 'result':result,'stderr_tail':(ART/'stderr.log').read_text()[-2000:] if p.returncode else '',
 'human_resource_authorization':lock['authorization'],'benchmark_references_read':False,'benchmark_test_scored':False,'elapsed_seconds':time.monotonic()-start}
(ART/'result.json').write_text(json.dumps(report,indent=2));(ROOT/'reports/QWEN_CAPTION_GPU_PILOT_V2_20261025.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report))
