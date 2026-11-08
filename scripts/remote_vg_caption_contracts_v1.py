"""CPU-only prompt/metric contract checks before finite VAL generation."""
from pathlib import Path
import gzip,json,os,hashlib,math
ROOT=Path.home()/'rsg-former';BASE=ROOT/'artifacts/vg_caption_execution_v1_20261025'
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
os.sched_setaffinity(0,{sorted(os.sched_getaffinity(0))[0]})
from transformers import AutoTokenizer
from rsgformer.vg_caption_prompt_v1 import bounded_query_graph,build_bank,make_messages,ARMS
from rsgformer.vg_caption_graph_v1 import serialize_graph
t=AutoTokenizer.from_pretrained(Path.home()/'etechs/msvd-tkg-captioning/models/qwen2.5-14b',local_files_only=True)
policy=json.loads((BASE/'color_policy_selection_v1.json').read_text())['caption_color_policy_for_VAL_exploratory_only']
with gzip.open(BASE/'color_candidates_v2/color_candidates.jsonl.gz','rt') as f:
 colors={r['key']:r.get('candidates',{}).get(policy['operating_point'],{}).get(policy['method'],[]) for r in map(json.loads,f)}
with gzip.open(BASE/'region_units_no_color.jsonl.gz','rt') as f:units=list(map(json.loads,f))
for u in units:
 for n in u['graph']['nodes']:n['accepted_colors']=colors.get(str(u['parent_image_id'])+':'+n['id'],[])
 u['graph']=bounded_query_graph(u['graph'],t)
blocked=json.loads((BASE/'color_gt_matches_v1/protected_train_bank_ids.json').read_text())
bank,static=build_bank([u for u in units if u['partition']=='train_bank'],t,blocked)
maximum=0;count=0;colored_units=0;empty=0
for u in [u for u in units if u['partition'] in ['caption_dev','caption_check']]:
 colored_units+=any(n.get('accepted_colors') for n in u['graph']['nodes']);empty+=not bool(u['graph']['nodes'])
 for template in ['P1','P2']:
  ids={}
  for arm in ARMS:
   messages,examples=make_messages(u,arm,template,bank,static)
   ids[arm]=examples
   tokens=t.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=False)
   maximum=max(maximum,len(tokens));assert len(tokens)<=768,(u['unit_id'],arm,len(tokens));count+=1
   if arm.startswith('N'):assert '"color":' not in messages[-1]['content']
  assert ids['N-D']==ids['C-D'] and ids['N-S']==ids['C-S'] and len(ids['N-S'])==len(ids['N-D'])==2
fixture={'a':['a man holding a phone'],'b':['a black cat on a chair'],'c':['a red shirt'],'d':['a car on the road']}
pred={'a':'a man holding a phone','b':'a cat on a chair','c':'','d':'a car'}
java=json.loads((ROOT/'reports/CAPTION_JAVA_PREPARATION.json').read_text())
j=Path(java['java_path_alias'].replace('${REMOTE_HOME}',str(Path.home())))
os.environ['PATH']=str(j.parent)+os.pathsep+os.environ['PATH']
os.environ['_JAVA_OPTIONS']='-Xmx2g -XX:ActiveProcessorCount=1 -XX:+UseSerialGC -XX:MaxMetaspaceSize=256m -XX:ReservedCodeCacheSize=128m'
from rsgformer.caption_metric_runtime import private_metric_runtime
from rsgformer.caption_metrics_v2 import compute_metrics
scratch=ROOT/'artifacts/caption_readiness/caption_execution_contracts_v1_20261025';scratch.mkdir(exist_ok=False)
with private_metric_runtime(ROOT,scratch) as runtime:
 metric=compute_metrics(fixture,pred,runtime,include_spice=False)
 # Compare wrapper with independent direct native complete-corpus METEOR.
 mods=runtime['modules'];tok=mods['PTBTokenizer'].PTBTokenizer()
 ids=sorted(fixture);gts=tok.tokenize({i:[{'caption':x} for x in fixture[k]] for i,k in enumerate(ids,1)})
 res=tok.tokenize({i:[{'caption':pred[k]}] for i,k in enumerate(ids,1)})
 scorer=mods['METEOR'].Meteor()
 try:value,rows=scorer.compute_score(gts,res)
 finally:
  if scorer.lock.locked():scorer.lock.release()
  del scorer
 assert abs(float(value)-metric['aggregate_native']['METEOR'])<=1e-12
 assert metric['evaluated_units']==4 and metric['empty_predictions']==1
report={'status':'PROMPT_AND_NATIVE_METRIC_CONTRACT_PASS','prompt_instances':count,'maximum_native_input_tokens':maximum,'TRAIN_examples':len(bank),
 'static_ids':static[:2],'VAL_dev_check_units':512,'units_with_candidate_color':colored_units,'empty_evidence_units':empty,
 'paired_retrieval_ids':True,'no_color_fields_in_N_prompts':True,'METEOR_native_parity_tolerance':1e-12,
 'metric_fixture':metric['aggregate_native'],'test_accessed':False}
(ROOT/'reports/VG_CAPTION_CONTRACTS_V1_20261025.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
