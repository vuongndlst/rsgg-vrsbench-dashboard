"""Exclusive CPU tokenizer/prompt preflight. No model generation or reference text."""
from pathlib import Path
import gzip,json,os,resource,signal,shutil,time,traceback
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUDA_CACHE_DISABLE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:4]));signal.alarm(600)
from rsgformer.vg_grounding_train_dev_runtime_v1_20261103 import *
from rsgformer.vg_caption_candidate_prompt_v2 import bounded
from rsgformer.vg_grounding_controls_v1_20261102 import synthetic_contract,validate_graph
from rsgformer.vg_caption_graph_v1 import digest
START=time.monotonic()
def main():
 assert not PREFLIGHT.exists(),'exclusive_preflight_no_retry'
 cfg,manifest=verify_inputs();assert shutil.disk_usage(ROOT).free>=17*1024**3 and artifact_bytes(ROOT/'artifacts')+1024**3<=40*1024**3
 assert int(next(s for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')).split()[1])*1024>=6*1024**3
 runtime=json.loads((ROOT/'configs/qwen_runtime.json').read_text());old_lock=json.loads((PARENT/'preregistration.json').read_text())
 for name in ['configs/qwen_runtime.json','src/rsgformer/qwen_streaming_v1.py','src/rsgformer/vg_caption_candidate_prompt_v2.py']:
  assert old_lock['source_sha256'][name]==manifest['source_sha256'][name],('G0_baseline_source_identity',name)
 folder=Path.home()/'etechs/msvd-tkg-captioning/models/qwen2.5-14b'
 for name,h in runtime['auxiliary_files_sha256'].items():assert sha(folder/name)==h
 from transformers import AutoTokenizer
 tok=AutoTokenizer.from_pretrained(folder,local_files_only=True)
 train=json.loads((PARENT/'train_generation/query_units_by_policy.json').read_text())['L2']
 with gzip.open(CACHE/'dev_units_unbounded_L2.jsonl.gz','rt') as f:dev=list(map(json.loads,f))
 baseline={r['unit_id']:r for r in map(json.loads,(PARENT/'train_generation/generation_commit.jsonl').read_text().splitlines()) if r['policy']=='L2'}
 assert len(train)==len(baseline)==800 and len(dev)==256
 assert not {u['parent_image_id'] for u in train}&{u['parent_image_id'] for u in dev}
 assert {u['parent_image_id'] for u in dev}==set(cfg['fresh_dev_ids'])
 max_tokens={'G0':0,'G2':0};prepared={};matched=0
 for split,units in [('train',train),('dev',dev)]:
  current=[]
  for original in units:
   u=dict(original);u['graph']=bounded(original['graph'],tok);validate_graph(u['graph'])
   if split=='train':assert u['graph']==original['graph']
   for arm,msg in prompt_rows(u).items():
    n=len(tok.apply_chat_template(msg,tokenize=True,add_generation_prompt=True,return_dict=False));assert n<=768,('native_input_cap',split,arm,u['unit_id'],n)
    max_tokens[arm]=max(max_tokens[arm],n)
    if split=='train' and arm=='G0':
     b=baseline[u['unit_id']];assert digest(u['graph'])==b['query_graph_sha256'] and digest(msg)==b['prompt_sha256'] and n==b['input_tokens'];matched+=1
   current.append(u)
  prepared[split]=current
 assert matched==800
 assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=4*1024**3
 assert time.monotonic()-START<600
 PREFLIGHT.mkdir()
 save(PREFLIGHT/'preregistration.json',{'utc':now(),'source_manifest_sha256':sha(MANIFEST),'config_sha256':sha(CFG),'source_sha256':manifest['source_sha256'],'input_hashes':{'parent':cfg['parent_files_sha256'],'dev':cfg['dev_cache_files_sha256']},'new_reference_text_read':False,'check_or_TEST_opened':False})
 for split,units in prepared.items():save(PREFLIGHT/('queries_'+split+'.json'),units)
 save(PREFLIGHT/'fixed20_dev_ids.json',sorted((u['unit_id'] for u in dev),key=lambda i:digest('VG_GROUNDING_DEV_FIXED20|'+i))[:20])
 result={'status':'GROUNDING_NATIVE_PROMPTS_AND_G0_REUSE_PREFLIGHT_PASS','utc':now(),'source_manifest_sha256':sha(MANIFEST),'config_sha256':sha(CFG),'synthetic_contract':synthetic_contract(),'TRAIN_units':800,'dev_units':256,'G0_TRAIN_exact_reuse_matches':matched,'maximum_native_input_tokens':max_tokens,'shared_query_graph_for_all_arms':True,'generator_reference_text_read':False,'new_inference_or_metric_scoring':False,'TEST_or_check_opened':False,'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'wall_seconds':time.monotonic()-START,'file_sha256':{p.name:sha(p) for p in PREFLIGHT.iterdir() if p.is_file()}}
 save(PREFLIGHT/'complete.json',result);assert artifact_bytes(PREFLIGHT)<64*1024**2
 print(line(result),flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:
  if PREFLIGHT.exists() and not (PREFLIGHT/'complete.json').exists():save(PREFLIGHT/'failure.json',{'utc':now(),'status':'FAILED_NO_AUTOMATIC_RETRY','reason':str(e)})
  traceback.print_exc();raise
