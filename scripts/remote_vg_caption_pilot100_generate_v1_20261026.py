"""100provided VAL regions,12frozen configurations; exact BF16 Qwen, no query refs."""
from pathlib import Path
import collections,gzip,json,os,resource,signal,time,traceback
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUDA_CACHE_DISABLE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
ROOT=Path.home()/'rsg-former';PIPE=ROOT/'artifacts/vg_caption_pilot100_v1_20261026';ART=PIPE/'generation';PREP=PIPE/'prepare';BASE=ROOT/'artifacts/vg_caption_execution_v1_20261025'
os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:4]));signal.alarm(7200)
from rsgformer.vg_color_execution_common_v1 import sha,save,line,now
from rsgformer.vg_caption_graph_v1 import digest
from rsgformer.vg_caption_candidate_prompt_v2 import ARMS,bounded,build_bank,messages

def main():
 assert not ART.exists();ART.mkdir()
 lock=json.loads((PIPE/'preregistration.json').read_text())
 for p,h in lock['source_sha256'].items():assert sha(ROOT/p)==h
 complete=json.loads((PREP/'complete.json').read_text());assert complete['pilot_images']==100
 for n,h in complete['file_sha256'].items():assert sha(PREP/n)==h
 import torch
 from transformers import AutoTokenizer
 from rsgformer.qwen_streaming_v1 import load_qwen
 torch.set_num_threads(4);torch.set_num_interop_threads(1)
 folder=Path.home()/'etechs/msvd-tkg-captioning/models/qwen2.5-14b'
 tokenizer=AutoTokenizer.from_pretrained(folder,local_files_only=True)
 with gzip.open(PREP/'pilot_units.jsonl.gz','rt') as f:units=sorted(map(json.loads,f),key=lambda u:u['unit_id'])
 with gzip.open(PREP/'train_units.jsonl.gz','rt') as f:train=list(map(json.loads,f))
 assert len(units)==len({u['parent_image_id'] for u in units})==100
 for u in units+train:u['graph']=bounded(u['graph'],tokenizer)
 blocked=json.loads((BASE/'color_gt_matches_v1/protected_train_bank_ids.json').read_text())
 bank,static=build_bank(train,tokenizer,blocked);save(ART/'train_demonstration_bank.json',{'examples':bank,'static_and_fallback_ids':static,'TRAIN_only':True,'human_annotation':False})
 save(ART/'query_units_bounded.json',units)
 maximum=0
 for u in units:
  for template in ['P1','P2']:
   for arm in ARMS:
    msg,ids=messages(u,arm,template,bank,static)
    tokens=tokenizer.apply_chat_template(msg,tokenize=True,add_generation_prompt=True,return_dict=False)
    assert len(tokens)<=768,('native_prompt_budget',u['unit_id'],arm,template,len(tokens));maximum=max(maximum,len(tokens))
    if arm.startswith('C'):
     _,other=messages(u,'N'+arm[1:],template,bank,static);assert other==ids,'paired_retrieval_ID_mismatch'
 save(ART/'prompt_contract.json',{'status':'ALL1200_PROMPTS_NATIVE_BUDGET_AND_PAIRED_IDS_PASS','max_input_tokens':maximum,'query_units':100,
  'bank_examples':len(bank),'static_ids':static,'original_query_reference_text_read':False,'test_accessed':False})
 save(ART/'preregistration.json',{'utc':now(),'parent_lock_sha256':sha(PIPE/'preregistration.json'),'preparation_complete_sha256':sha(PREP/'complete.json'),
  'bank_sha256':sha(ART/'train_demonstration_bank.json'),'query_units_sha256':sha(ART/'query_units_bounded.json'),
  'model_runtime_sha256':sha(ROOT/'configs/qwen_runtime.json'),'color_semantics':'unverified_candidate_only','TEST_access_allowed':False})
 save(ART/'progress.json',{'utc':now(),'stage':'loading_exact_BF16_Qwen','expected_outputs':1200})
 model,tokenizer=load_qwen(folder,verify_weights=True);count=0;start=time.monotonic()
 with (ART/'generation_commit.jsonl').open('x') as out:
  for template in ['P1','P2']:
   for arm in ARMS:
    for u in units:
     assert not (PIPE/'STOP').exists() and time.time()<1794670321.2414355
     msg,examples=messages(u,arm,template,bank,static)
     inputs=tokenizer.apply_chat_template(msg,tokenize=True,add_generation_prompt=True,return_tensors='pt',return_dict=True).to('cuda')
     caption='';new=[];seconds=0.
     if u['graph']['nodes']:
      at=time.monotonic()
      with torch.inference_mode():generated=model.generate(**inputs,do_sample=False,num_beams=1,max_new_tokens=32,pad_token_id=tokenizer.eos_token_id,use_cache=True)
      new=generated[0,inputs['input_ids'].shape[-1]:].cpu().tolist();caption=tokenizer.decode(new,skip_special_tokens=True).strip();seconds=time.monotonic()-at
     row={'unit_id':u['unit_id'],'parent_image_id':u['parent_image_id'],'arm':arm,'template':template,'caption':caption,'new_token_ids':new,
      'input_tokens':int(inputs['input_ids'].shape[-1]),'example_ids':examples,'prompt_sha256':digest(msg),'query_graph_sha256':digest(u['graph']),
      'no_evidence':not bool(u['graph']['nodes']),'hit_token_cap':len(new)==32,'seconds':seconds,'color_scope':'unverified_candidate_only' if arm.startswith('C') else 'no_color'}
     out.write(line(row)+'\n');out.flush();count+=1;del inputs
     if count%25==0:save(ART/'progress.json',{'utc':now(),'stage':'generation','outputs':count,'expected_outputs':1200,'arm':arm,'template':template})
 assert count==1200
 result={'status':'PILOT100_TWELVE_CONFIG_CAPTIONS_COMPLETE','utc':now(),'images':100,'provided_regions':100,'outputs':count,
  'generation_sha256':sha(ART/'generation_commit.jsonl'),'bank_sha256':sha(ART/'train_demonstration_bank.json'),'query_units_sha256':sha(ART/'query_units_bounded.json'),
  'peak_allocator_bytes':torch.cuda.max_memory_allocated(),'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
  'seconds_after_load':time.monotonic()-start,'TEST_accessed':False,'query_reference_text_read':False,'human_color_precision_gate_passed':False}
 save(ART/'complete.json',result);print(line(result),flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:
  if ART.exists():save(ART/'failure.json',{'utc':now(),'reason':str(e),'TEST_accessed':False})
  traceback.print_exc();raise
