"""Six frozen finite VAL ablations. Candidate-color arms exploratory; TEST forbidden."""
from pathlib import Path
from datetime import datetime,timezone
import gc,gzip,hashlib,json,os,resource,shutil,subprocess,time,traceback
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUDA_CACHE_DISABLE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
ROOT=Path.home()/'rsg-former';BASE=ROOT/'artifacts/vg_caption_execution_v1_20261025';ART=BASE/'caption_check_v1';START=time.time()
os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:4]))
from rsgformer.vg_caption_graph_v1 import digest,serialize_graph
from rsgformer.vg_caption_prompt_v1 import bounded_query_graph,build_bank,make_messages,ARMS

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def save(p,x):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2));t.replace(p)
def guard():
 assert time.time()<1794670321.2414355 and time.time()-START<4*3600,'wall_deadline'
 assert shutil.disk_usage(ROOT).free>=16*1024**3,'disk_reserve16GiB'
 assert not (BASE/'STOP').exists(),'owned_stop'
 assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=12*1024**3,'rss12GiB'
 assert sum(p.stat().st_size for p in BASE.rglob('*') if p.is_file())<=8*1024**3,'phase_artifacts8GiB'
def event(stage,**kw):
 guard();save(ART/'progress.json',{'utc':datetime.now(timezone.utc).isoformat(),'stage':stage,**kw})
def main():
 assert (BASE/'color_policy_selection_v1.json').exists() and (BASE/'caption_prompt_selection_v1.json').exists() and not ART.exists()
 pilot=json.loads((ROOT/'reports/QWEN_CAPTION_GPU_PILOT_V2_20261025.json').read_text());assert pilot['status']=='QWEN_BF16_SYNTHETIC_GPU_PILOT_PASS'
 guard();ART.mkdir()
 lock={'phase':'caption_check_v1','created_utc':datetime.now(timezone.utc).isoformat(),'source_hashes':{p:sha(ROOT/p) for p in [
  'scripts/remote_vg_caption_generation_check_v1.py','src/rsgformer/qwen_streaming_v1.py','src/rsgformer/vg_caption_prompt_v1.py','src/rsgformer/vg_caption_graph_v1.py']},
  'input_hashes':{p:sha(BASE/p) for p in ['region_units_no_color.jsonl.gz','color_policy_selection_v1.json','membership.json','caption_prompt_selection_v1.json','caption_dev_v1/train_demonstration_bank.json','color_candidates_v2/color_candidates.jsonl.gz']},
  'arms':ARMS,'templates':[json.loads((BASE/'caption_prompt_selection_v1.json').read_text())['selected_template']],'partition':'caption_check','images':64,'maximum_units':256,'maximum_generations':1536,
  'maximum_input_tokens':768,'maximum_output_tokens':32,'do_sample':False,'num_beams':1,'fewshot_k':2,'query_base_budget':320,'query_color_budget':448,
  'exemplar_budget':256,'resource':{'GPU_GiB':36,'allocator_GiB':32,'RSS_GiB':12,'CPU':4},
  'qwen_runtime_config_sha256':sha(ROOT/'configs/qwen_runtime.json'),
  'no_evidence':'empty output retained in all arm denominators, no GT fallback',
  'examples':'paired curated predicted-fact templates from TRAIN, not original human-reference answers',
  'color_arm_scope':'thresholded candidate color on VAL exploratory only; human visual gate pending, cannot authorize color TEST',
  'selection':'template already locked on dev; held-out check cannot change selection or thresholds',
  'benchmark_TEST_access_allowed':False,'new_SGDet_test_scoring':False}
 save(ART/'preregistration.json',lock);event('loading_exact_BF16_Qwen_streaming')
 import torch
 from transformers import AutoTokenizer
 from rsgformer.qwen_streaming_v1 import load_qwen
 torch.set_num_threads(4);torch.set_num_interop_threads(1)
 assert torch.cuda.mem_get_info()[0]>=38*1024**3,'free_GPU38GiB'
 model,tokenizer=load_qwen(Path.home()/'etechs/msvd-tkg-captioning/models/qwen2.5-14b',verify_weights=True)
 policy=json.loads((BASE/'color_policy_selection_v1.json').read_text())['caption_color_policy_for_VAL_exploratory_only']
 with gzip.open(BASE/'color_candidates_v2/color_candidates.jsonl.gz','rt') as f:
  colors={r['key']:r.get('candidates',{}).get(policy['operating_point'],{}).get(policy['method'],[]) for r in map(json.loads,f)}
 with gzip.open(BASE/'region_units_no_color.jsonl.gz','rt') as f:all_units=list(map(json.loads,f))
 # Model inputs are loaded from the reference-free cache. TRAIN reference hash is provenance only.
 for u in all_units:
  assert u['partition'] in ['train_bank','caption_dev','caption_check','color_dev','color_check']
  for n in u['graph']['nodes']:
   n['accepted_colors']=colors.get(str(u['parent_image_id'])+':'+n['id'],[])
  u['graph']=bounded_query_graph(u['graph'],tokenizer)
 blocked=json.loads((BASE/'color_gt_matches_v1/protected_train_bank_ids.json').read_text())
 locked_bank=json.loads((BASE/'caption_dev_v1/train_demonstration_bank.json').read_text());bank=locked_bank['examples'];static=locked_bank['static_and_fallback_ids']
 save(ART/'train_demonstration_bank.json',{'examples':bank,'static_and_fallback_ids':static,'provenance':'TRAIN-only curated predicted-fact templates; no query references read'})
 units=sorted([u for u in all_units if u['partition']=='caption_check'],key=lambda u:u['unit_id']);assert len(units)==256
 # Validate all prompt budgets before committing generation outputs.
 for u in units:
  for arm in ARMS:
   for template in lock['templates']:
    messages,_=make_messages(u,arm,template,bank,static)
    tokens=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=False)
    assert len(tokens)<=768,'prompt_budget_contract_no_implicit_truncation'
 event('generating_six_VAL_arms',units=len(units),bank_examples=len(bank))
 count=0
 with (ART/'generation_commit.jsonl').open('x') as out:
  for template in lock['templates']:
   for arm in ARMS:
    for u in units:
     guard();messages,example_ids=make_messages(u,arm,template,bank,static);text='';new_tokens=[];seconds=0
     prompt_ids=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_tensors='pt',return_dict=True).to('cuda')
     if u['graph']['nodes']:
      start=time.monotonic()
      with torch.inference_mode():generated=model.generate(**prompt_ids,do_sample=False,num_beams=1,max_new_tokens=32,pad_token_id=tokenizer.eos_token_id,use_cache=True)
      new_tokens=generated[0,prompt_ids['input_ids'].shape[-1]:].cpu().tolist();text=tokenizer.decode(new_tokens,skip_special_tokens=True).strip();seconds=time.monotonic()-start
     row={'unit_id':u['unit_id'],'parent_image_id':u['parent_image_id'],'partition':'caption_check','arm':arm,'template':template,
      'caption':text,'new_token_ids':new_tokens,'prompt_sha256':digest(messages),'example_ids':example_ids,'query_graph_sha256':digest(u['graph']),
      'input_tokens':int(prompt_ids['input_ids'].shape[-1]),'generation_seconds':seconds,'no_evidence':not bool(u['graph']['nodes']),
      'color_semantics':'VAL_candidate_only' if arm.startswith('C') else 'no_color','hit_token_cap':len(new_tokens)==32}
     out.write(json.dumps(row)+'\n');out.flush();count+=1
     if count%64==0:event('VAL_generation',committed_outputs=count,maximum=1536,template=template,arm=arm,peak_allocator_bytes=torch.cuda.max_memory_allocated())
     del prompt_ids
    event('VAL_arm_complete',arm=arm,template=template,committed_outputs=count)
 result={'status':'SIX_ARM_LOCKED_TEMPLATE_VAL_CHECK_GENERATION_COMPLETE','utc':datetime.now(timezone.utc).isoformat(),'outputs':count,'units':len(units),
  'generation_sha256':sha(ART/'generation_commit.jsonl'),'bank_sha256':sha(ART/'train_demonstration_bank.json'),
  'peak_allocator_bytes':torch.cuda.max_memory_allocated(),'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
  'test_accessed':False,'dataset_reference_text_read_by_generator':False,'wall_seconds':time.time()-START}
 save(ART/'complete.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':
 try:main()
 except Exception as e:
  if ART.exists():save(ART/'failure.json',{'utc':datetime.now(timezone.utc).isoformat(),'error_type':type(e).__name__,'reason':str(e),'test_accessed':False})
  traceback.print_exc();raise
