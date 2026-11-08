"""TRAIN-only selection or one frozen heldout check; full corpus, no TEST selection."""
from pathlib import Path
import collections,json,os,re,signal,traceback
ROOT=Path.home()/'rsg-former';PIPE=ROOT/'artifacts/vg_focus_repair_v1_20261028';BASE=ROOT/'artifacts/vg_caption_execution_v1_20261025';SPLIT=os.environ['VG_FOCUS_STAGE'];assert SPLIT in ['train','check'];ART=PIPE/(SPLIT+'_metrics');GEN=PIPE/(SPLIT+'_generation')
os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:4]));signal.alarm(2700)
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
from rsgformer.vg_color_execution_common_v1 import sha,save,line,now

def main():
 assert not ART.exists();ART.mkdir();lock=json.loads((PIPE/'preregistration.json').read_text())
 for p,h in lock['source_sha256'].items():assert sha(ROOT/p)==h
 gen=json.loads((GEN/'complete.json').read_text());assert sha(GEN/'generation_commit.jsonl')==gen['generation_sha256'] and sha(GEN/'query_units_by_policy.json')==gen['query_units_sha256']
 policies=gen['policies'];queries=json.loads((GEN/'query_units_by_policy.json').read_text());ids={u['unit_id'] for u in queries['L0']};assert len(ids)==(800 if SPLIT=='train' else 256)
 original=json.loads((BASE/'base_cache_complete.json').read_text());assert sha(BASE/'human_references_separate.json')==original['file_sha256']['human_references_separate.json']
 all_refs=json.loads((BASE/'human_references_separate.json').read_text());refs={i:all_refs[i] for i in ids};save(ART/'original_human_references.json',refs)
 save(ART/'preregistration.json',{'utc':now(),'source_lock_sha256':sha(PIPE/'preregistration.json'),'split':SPLIT,'reference_source_sha256':sha(BASE/'human_references_separate.json'),
  'generation_sha256':gen['generation_sha256'],'scored_unit_ids':sorted(ids),'reference_scope':'shared original reference file read by scorer, only this split subset used','full_corpus_CIDEr_global_IDF':True,
  'SPICE':'not_requested_in_focus_diagnostic','selection_allowed':SPLIT=='train','TEST_access_allowed':False})
 java=json.loads((ROOT/'reports/CAPTION_JAVA_PREPARATION.json').read_text());j=Path(java['java_path_alias'].replace('${REMOTE_HOME}',str(Path.home())))
 os.environ['PATH']=str(j.parent)+os.pathsep+os.environ['PATH'];os.environ['_JAVA_OPTIONS']='-Xmx2g -XX:ActiveProcessorCount=1 -XX:+UseSerialGC -XX:MaxMetaspaceSize=256m -XX:ReservedCodeCacheSize=128m'
 from rsgformer.caption_metric_runtime import private_metric_runtime
 from rsgformer.caption_metrics_v2 import compute_metrics
 rows=list(map(json.loads,(GEN/'generation_commit.jsonl').read_text().splitlines()));assert len(rows)==len(ids)*len(policies)
 results={};colorwords=set('black white gray grey red green blue yellow brown orange pink purple'.split())
 for policy in policies:
  local=[r for r in rows if r['policy']==policy];assert len(local)==len(ids) and {r['unit_id'] for r in local}==ids
  scratch=ROOT/'artifacts/caption_readiness'/('focus_repair_20261028_'+SPLIT+'_'+policy);scratch.mkdir(exist_ok=False)
  with private_metric_runtime(ROOT,scratch) as runtime:metric=compute_metrics(refs,{r['unit_id']:r['caption'] for r in local},runtime,include_spice=False)
  q={u['unit_id']:u['graph'] for u in queries[policy]};colors=counts=0
  for r in local:
   words=set(re.findall('[a-z]+',r['caption'].lower()));object_words={n['label'] for n in q[r['unit_id']]['nodes']}
   colors+=bool((words&colorwords)-object_words)
   counts+=bool(re.search(r'\b(two|three|four|five|six|seven|eight|nine|ten|[2-9][0-9]*)\b',r['caption'].lower()))
  metric.update(policy=policy,split=SPLIT,empty_evidence_units=sum(r['empty_evidence'] for r in local),output_token_cap_hits=sum(r['hit_token_cap'] for r in local),
   unsupported_color_lexical_flags=colors,unsupported_count_lexical_flags=counts,factuality_audit_status='AUTOMATED_PROXY_NOT_HUMAN_ACCURACY',SPICE_status='NOT_RUN_FOCUS_DIAGNOSTIC')
  save(ART/(policy+'.json'),metric);results[policy]=metric;save(ART/'progress.json',{'utc':now(),'split':SPLIT,'corpora':len(results),'expected':len(policies)})
 selected=max(policies,key=lambda p:results[p]['aggregate_native']['METEOR']) if SPLIT=='train' else json.loads((PIPE/'selection.json').read_text())['selected_policy']
 report={'status':'FOCUS_REPAIR_'+SPLIT.upper()+'_METRICS_COMPLETE','utc':now(),'split':SPLIT,'selected_policy':selected,'selected_from_TRAIN_only':True,
  'selection_rule':'full-corpus TRAIN METEOR, exact ties L0 thenL1 thenL2','units_per_policy':len(ids),'policies':policies,'native_scores':{p:m['aggregate_native'] for p,m in results.items()},
  'display_scores_x100':{p:m['aggregate_display_x100'] for p,m in results.items()},
  'diagnostics':{p:{k:m[k] for k in ['empty_predictions','empty_evidence_units','output_token_cap_hits','unsupported_color_lexical_flags','unsupported_count_lexical_flags']} for p,m in results.items()},
  'SPICE_status':'NOT_RUN_FOCUS_DIAGNOSTIC','policy_adopted':False,'semantic_correctness_established':False,'automatic_TEST_launch':False,
  'metric_sha256':{p:sha(ART/(p+'.json')) for p in policies},'generation_sha256':gen['generation_sha256']}
 if SPLIT=='check':
  assert (PIPE/'check_access.json').exists();sample=json.loads((PIPE/'prepare/fixed_check_qualitative_sample_ids.json').read_text())
  save(ART/'fixed20_factuality_review_packet.json',[{'unit_id':i,'reference':refs[i],'outputs':{r['policy']:r['caption'] for r in rows if r['unit_id']==i},'query_graphs':{p:next(u['graph'] for u in queries[p] if u['unit_id']==i) for p in policies},'human_review_completed':False} for i in sample])
  report['fixed_qualitative_sample_count']=20;report['independent_human_factuality_review_complete']=False
  report['check_METEOR_difference_native']=results[selected]['aggregate_native']['METEOR']-results['L0']['aggregate_native']['METEOR']
 save(ART/'complete.json',report)
 if SPLIT=='train':
  selection={'utc':now(),'status':'POLICY_LOCKED_FROM_TRAIN_FOR_HELDOUT_COMPARISON_ONLY','selected_policy':selected,'TRAIN_metrics_completion_sha256':sha(ART/'complete.json'),
   'source_lock_sha256':sha(PIPE/'preregistration.json'),'selection_uses_check_or_TEST':False,'policy_adopted':False}
  with (PIPE/'selection.json').open('x') as f:f.write(line(selection)+'\n')
 save(ROOT/('reports/VG_FOCUS_REPAIR_'+SPLIT.upper()+'_RESULT_20261028.json'),report);print(line(report),flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:
  if ART.exists():save(ART/'failure.json',{'utc':now(),'reason':str(e),'split':SPLIT,'TEST_accessed':False})
  traceback.print_exc();raise
