"""Finite no-color zero-shot P1 generation; check requires immutable TRAIN selection."""
from pathlib import Path
import gzip,json,os,signal,time,traceback,resource
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUDA_CACHE_DISABLE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
ROOT=Path.home()/'rsg-former';PIPE=ROOT/'artifacts/vg_focus_repair_v1_20261028';SPLIT=os.environ['VG_FOCUS_STAGE'];assert SPLIT in ['train','check'];ART=PIPE/(SPLIT+'_generation');PREP=PIPE/'prepare'
os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:4]));signal.alarm(7200)
from rsgformer.vg_color_execution_common_v1 import sha,save,line,now
from rsgformer.vg_caption_graph_v1 import digest
from rsgformer.vg_region_focus_repair_v1 import select,POLICIES
from rsgformer.vg_caption_candidate_prompt_v2 import bounded,messages

def main():
 assert not ART.exists();ART.mkdir();lock=json.loads((PIPE/'preregistration.json').read_text())
 for p,h in lock['source_sha256'].items():assert sha(ROOT/p)==h
 complete=json.loads((PREP/'complete.json').read_text())
 for n in [SPLIT+'_units.jsonl.gz',SPLIT+'_graphs.jsonl.gz']:assert sha(PREP/n)==complete['file_sha256'][n]
 policies=POLICIES;selection=None
 if SPLIT=='check':
  assert not (PIPE/'check_access.json').exists()
  selection=json.loads((PIPE/'selection.json').read_text());train_metric=json.loads((PIPE/'train_metrics/complete.json').read_text())
  assert sha(PIPE/'train_metrics/complete.json')==selection['TRAIN_metrics_completion_sha256'] and train_metric['selected_policy']==selection['selected_policy']
  policies=list(dict.fromkeys(['L0',selection['selected_policy']]))
 with gzip.open(PREP/(SPLIT+'_units.jsonl.gz'),'rt') as f:units=list(map(json.loads,f))
 with gzip.open(PREP/(SPLIT+'_graphs.jsonl.gz'),'rt') as f:graphs={r['image_id']:r['graph'] for r in map(json.loads,f)}
 assert len(units)==(800 if SPLIT=='train' else 256)
 import torch
 from transformers import AutoTokenizer
 from rsgformer.qwen_streaming_v1 import load_qwen
 torch.set_num_threads(4);torch.set_num_interop_threads(1)
 folder=Path.home()/'etechs/msvd-tkg-captioning/models/qwen2.5-14b';tokenizer=AutoTokenizer.from_pretrained(folder,local_files_only=True)
 materialized={};maximum=0
 for policy in policies:
  current=[]
  for u in units:
   q=dict(u);q['graph']=bounded(select(graphs[u['parent_image_id']],u['crop_xyxy'],policy),tokenizer)
   assert not any(n.get('candidate_colors') or n.get('accepted_colors') for n in q['graph']['nodes'])
   msg,examples=messages(q,'N-Z','P1',[],[]);assert not examples
   n=len(tokenizer.apply_chat_template(msg,tokenize=True,add_generation_prompt=True,return_dict=False));assert n<=768;maximum=max(maximum,n);current.append(q)
  materialized[policy]=current
 save(ART/'query_units_by_policy.json',materialized)
 save(ART/'prompt_contract.json',{'status':'ALL_NATIVE_PROMPTS_PASS','policies':policies,'maximum_input_tokens':maximum,'no_fewshot_or_color':True,'query_reference_text_read':False})
 save(ART/'preregistration.json',{'utc':now(),'parent_lock_sha256':sha(PIPE/'preregistration.json'),'prepared_completion_sha256':sha(PREP/'complete.json'),
  'split':SPLIT,'policies':policies,'units_per_policy':len(units),'template':'frozenP1_N-Z','maximum_new_tokens':32,'query_units_sha256':sha(ART/'query_units_by_policy.json'),
  'selection_sha256':sha(PIPE/'selection.json') if selection else None,'no_GT_reference_in_generator':True,'TEST_access_allowed':False})
 if SPLIT=='check':
  with (PIPE/'check_access.json').open('x') as f:f.write(line({'utc':now(),'phase':'heldout_caption_check_once','selected_from_TRAIN_only':selection['selected_policy'],'selection_sha256':sha(PIPE/'selection.json'),'TEST_accessed':False})+'\n');f.flush();os.fsync(f.fileno())
 save(ART/'progress.json',{'utc':now(),'stage':'loading_exact_BF16_Qwen','split':SPLIT})
 model,tokenizer=load_qwen(folder,verify_weights=True);count=0;start=time.monotonic()
 with (ART/'generation_commit.jsonl').open('x') as out:
  for policy in policies:
   for u in materialized[policy]:
    assert not (PIPE/'STOP').exists() and time.time()<1794670321.2414355
    msg,_=messages(u,'N-Z','P1',[],[]);inp=tokenizer.apply_chat_template(msg,tokenize=True,add_generation_prompt=True,return_tensors='pt',return_dict=True).to('cuda')
    text='';new=[]
    if u['graph']['nodes']:
     with torch.inference_mode():generated=model.generate(**inp,do_sample=False,num_beams=1,max_new_tokens=32,pad_token_id=tokenizer.eos_token_id,use_cache=True)
     new=generated[0,inp['input_ids'].shape[-1]:].cpu().tolist();text=tokenizer.decode(new,skip_special_tokens=True).strip()
    out.write(line({'unit_id':u['unit_id'],'parent_image_id':u['parent_image_id'],'split':SPLIT,'policy':policy,'caption':text,'new_token_ids':new,'input_tokens':int(inp['input_ids'].shape[-1]),
     'prompt_sha256':digest(msg),'query_graph_sha256':digest(u['graph']),'empty_evidence':not bool(u['graph']['nodes']),'hit_token_cap':len(new)==32})+'\n');out.flush();count+=1;del inp
    if count%50==0:save(ART/'progress.json',{'utc':now(),'split':SPLIT,'outputs':count,'expected':len(units)*len(policies),'policy':policy})
 assert count==len(units)*len(policies)
 result={'status':'FOCUS_REPAIR_'+SPLIT.upper()+'_GENERATION_COMPLETE','utc':now(),'split':SPLIT,'policies':policies,'units_per_policy':len(units),'outputs':count,
  'generation_sha256':sha(ART/'generation_commit.jsonl'),'query_units_sha256':sha(ART/'query_units_by_policy.json'),'peak_allocator_bytes':torch.cuda.max_memory_allocated(),
  'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'seconds_after_load':time.monotonic()-start,'query_reference_text_read':False,'TEST_accessed':False}
 save(ART/'complete.json',result);print(line(result),flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:
  if ART.exists():save(ART/'failure.json',{'utc':now(),'reason':str(e),'split':SPLIT,'TEST_accessed':False})
  traceback.print_exc();raise
