"""Finite TRAIN/dev grounding generation; no reference text or check/TEST access."""
from pathlib import Path
import json,os,resource,signal,time,traceback
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUDA_CACHE_DISABLE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:4]));signal.alarm(7200)
from rsgformer.vg_grounding_train_dev_runtime_r1_20261103 import *
from rsgformer.vg_caption_graph_v1 import digest
from rsgformer.vg_grounding_controls_v1_20261102 import literal_first,parse_and_render
SPLIT=os.environ['VG_GROUNDING_SPLIT'];assert SPLIT in ['train','dev'];ART=PIPE/(SPLIT+'_generation')
def main():
 assert not ART.exists();cfg,manifest=verify_inputs();lock=json.loads((PIPE/'preregistration.json').read_text());pre=json.loads((PREFLIGHT/'complete.json').read_text())
 assert sha(PREFLIGHT/'complete.json')==lock['preflight_completion_sha256']
 qpath=PREFLIGHT/('queries_'+SPLIT+'.json');assert sha(qpath)==pre['file_sha256'][qpath.name];units=json.loads(qpath.read_text());assert len(units)==(800 if SPLIT=='train' else 256)
 ART.mkdir();save(ART/'queries.json',units)
 save(ART/'preregistration.json',{'utc':now(),'parent_lock_sha256':sha(PIPE/'preregistration.json'),'preflight_sha256':sha(PREFLIGHT/'complete.json'),'split':SPLIT,'arms':['G0','G1','G2'],'units_per_arm':len(units),'query_sha256':sha(ART/'queries.json'),'G0_TRAIN_reuse_only':SPLIT=='train','max_new_tokens':32,'new_reference_text_read':False,'check_or_TEST_access_allowed':False})
 import torch
 from rsgformer.qwen_streaming_v1 import load_qwen
 torch.set_num_threads(4);torch.set_num_interop_threads(1)
 model,tok=load_qwen(Path.home()/'etechs/msvd-tkg-captioning/models/qwen2.5-14b',verify_weights=True)
 baseline={r['unit_id']:r for r in map(json.loads,(PARENT/'train_generation/generation_commit.jsonl').read_text().splitlines()) if r['policy']=='L2'} if SPLIT=='train' else {}
 count=calls=reused=invalid=0;max_input=0;start=time.monotonic()
 with (ART/'generation_commit.jsonl').open('x') as stream:
  for arm in ['G0','G1','G2']:
   for u in units:
    assert time.time()<1794670321.2414355 and not (PIPE/'STOP').exists()
    empty=not bool(u['graph']['nodes']);new=[];raw=None;choice=None;prompt_hash=None;inputs=0;status='free_form_generated'
    if arm=='G1':
     realized=literal_first(u['graph']);text=realized['caption'];status=realized['status'];choice=realized['choice']
    elif arm=='G0' and SPLIT=='train':
     b=baseline[u['unit_id']];assert b['query_graph_sha256']==digest(u['graph']);text=b['caption'];new=b['new_token_ids'];inputs=b['input_tokens'];prompt_hash=b['prompt_sha256'];status='byte_identical_parent_G0_TRAIN_caption_reused';reused+=1
    else:
     msg=prompt_rows(u)[arm];prompt_hash=digest(msg)
     inp=tok.apply_chat_template(msg,tokenize=True,add_generation_prompt=True,return_tensors='pt',return_dict=True).to('cuda');inputs=int(inp['input_ids'].shape[-1]);assert inputs<=768
     text=''
     if not empty:
      with torch.inference_mode():generated=model.generate(**inp,do_sample=False,num_beams=1,max_new_tokens=32,pad_token_id=tok.eos_token_id,use_cache=True)
      new=generated[0,inputs:].cpu().tolist();text=tok.decode(new,skip_special_tokens=True).strip();calls+=1;del generated
     del inp
     if arm=='G2':
      raw=text;realized=parse_and_render(raw,u['graph']);text=realized['caption'];status=realized['status'];choice=realized['choice'];invalid+=status=='invalid_selection_abstention'
     elif empty:status='empty_evidence_abstention'
    max_input=max(max_input,inputs)
    stream.write(line({'unit_id':u['unit_id'],'parent_image_id':u['parent_image_id'],'split':SPLIT,'arm':arm,'caption':text,'raw_selection_output':raw,'selection':choice,'status':status,'new_token_ids':new,'input_tokens':inputs,'prompt_sha256':prompt_hash,'query_graph_sha256':digest(u['graph']),'empty_evidence':empty,'hit_token_cap':len(new)==32,'parent_reused':arm=='G0' and SPLIT=='train'})+'\n');stream.flush();count+=1
    if count%50==0:save(ART/'progress.json',{'utc':now(),'split':SPLIT,'records':count,'expected_records':len(units)*3,'arm':arm,'new_model_calls':calls,'invalid_selection_abstentions':invalid})
 assert count==len(units)*3 and calls<=(800 if SPLIT=='train' else 512) and reused==(800 if SPLIT=='train' else 0)
 result={'status':'GROUNDING_'+SPLIT.upper()+'_GENERATION_COMPLETE','utc':now(),'split':SPLIT,'arms':['G0','G1','G2'],'units_per_arm':len(units),'records':count,'new_model_calls':calls,'G0_TRAIN_reused':reused,'G2_invalid_selection_abstentions':invalid,'generation_sha256':sha(ART/'generation_commit.jsonl'),'query_sha256':sha(ART/'queries.json'),'maximum_input_tokens':max_input,'peak_allocator_bytes':torch.cuda.max_memory_allocated(),'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'seconds_after_load':time.monotonic()-start,'generator_reference_text_read':False,'check_or_TEST_opened':False}
 save(ART/'complete.json',result);print(line(result),flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:
  if ART.exists():save(ART/'failure.json',{'utc':now(),'reason':str(e),'status':'FAILED_NO_AUTOMATIC_RETRY','split':SPLIT})
  traceback.print_exc();raise
