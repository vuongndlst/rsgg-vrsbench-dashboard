"""Full TRAIN/dev native metrics, TRAIN G0 score reuse, dev-only selection."""
from pathlib import Path
import json,os,signal,traceback
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1');os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:4]));signal.alarm(2700)
from rsgformer.vg_grounding_train_dev_runtime_v1_20261103 import *
from rsgformer.vg_grounding_controls_v1_20261102 import literal_first,parse_and_render
from rsgformer.vg_caption_graph_v1 import digest
SPLIT=os.environ['VG_GROUNDING_SPLIT'];assert SPLIT in ['train','dev'];ART=PIPE/(SPLIT+'_metrics');GEN=PIPE/(SPLIT+'_generation')
def main():
 assert not ART.exists();cfg,manifest=verify_inputs();gen=json.loads((GEN/'complete.json').read_text())
 assert sha(GEN/'generation_commit.jsonl')==gen['generation_sha256'] and sha(GEN/'queries.json')==gen['query_sha256']
 units=json.loads((GEN/'queries.json').read_text());queries={u['unit_id']:u['graph'] for u in units};ids=set(queries);expected=800 if SPLIT=='train' else 256
 assert len(ids)==expected
 rows=list(map(json.loads,(GEN/'generation_commit.jsonl').read_text().splitlines()));assert len(rows)==expected*3
 assert len({(r['arm'],r['unit_id']) for r in rows})==expected*3
 for r in rows:
  assert digest(queries[r['unit_id']])==r['query_graph_sha256']
  if r['arm']=='G1':assert literal_first(queries[r['unit_id']])['caption']==r['caption']
  if r['arm']=='G2':assert parse_and_render(r['raw_selection_output'],queries[r['unit_id']])['caption']==r['caption']
 refpath=PARENT/'train_metrics/original_human_references.json' if SPLIT=='train' else CACHE/'dev_references_scoring_only.json'
 refs=json.loads(refpath.read_text());assert set(refs)==ids and all(isinstance(v,list) and v for v in refs.values()),'no_missing_reference_drop_or_fabrication'
 ART.mkdir();save(ART/'original_human_references.json',refs)
 save(ART/'preregistration.json',{'utc':now(),'source_manifest_sha256':sha(MANIFEST),'source_lock_sha256':sha(PIPE/'preregistration.json'),'split':SPLIT,'reference_source_sha256':sha(refpath),'generation_sha256':gen['generation_sha256'],'expected_units':expected,'full_corpus_CIDEr_global_IDF':True,'SPICE':'NOT_RUN','selection_allowed':SPLIT=='dev','check_or_TEST_accessed':False})
 java=json.loads((ROOT/'reports/CAPTION_JAVA_PREPARATION.json').read_text());j=Path(java['java_path_alias'].replace('${REMOTE_HOME}',str(Path.home())))
 os.environ['PATH']=str(j.parent)+os.pathsep+os.environ['PATH'];os.environ['_JAVA_OPTIONS']='-Xmx2g -XX:ActiveProcessorCount=1 -XX:+UseSerialGC -XX:MaxMetaspaceSize=256m -XX:ReservedCodeCacheSize=128m'
 from rsgformer.caption_metric_runtime import private_metric_runtime
 from rsgformer.caption_metrics_v2 import compute_metrics
 original_metric=json.loads((PARENT/'train_metrics/L2.json').read_text());results={}
 for arm in ['G0','G1','G2']:
  local=[r for r in rows if r['arm']==arm];assert len(local)==expected and {r['unit_id'] for r in local}==ids
  if SPLIT=='train' and arm=='G0':
   assert set(original_metric['unit_ids'])==ids and original_metric['evaluated_units']==original_metric['expected_units']==expected
   metric=dict(original_metric);(ART/'G0_original_parent_L2_metric.json').write_bytes((PARENT/'train_metrics/L2.json').read_bytes());reused=True
  else:
   scratch=ROOT/'artifacts/caption_readiness'/('grounding_train_dev_v1_20261103_'+SPLIT+'_'+arm);scratch.mkdir(exist_ok=False)
   with private_metric_runtime(ROOT,scratch) as runtime:metric=compute_metrics(refs,{r['unit_id']:r['caption'] for r in local},runtime,include_spice=False)
   assert metric['metric_sources']==original_metric['metric_sources'],'native_metric_library_drift';reused=False
  metric.update(arm=arm,split=SPLIT,G0_TRAIN_original_metric_reused_not_rescored=reused,original_G0_TRAIN_metric_sha256=sha(PARENT/'train_metrics/L2.json') if reused else None,
   G2_invalid_selection_abstentions=sum(r['status']=='invalid_selection_abstention' for r in local),empty_evidence_units=sum(r['empty_evidence'] for r in local),output_token_cap_hits=sum(r['hit_token_cap'] for r in local),
   graph_support_is_structural_not_human_accuracy=True,SPICE_status='NOT_RUN_GROUNDING_DIAGNOSTIC')
  save(ART/(arm+'.json'),metric);results[arm]=metric;save(ART/'progress.json',{'utc':now(),'split':SPLIT,'corpora':len(results),'expected':3})
 result={'status':'GROUNDING_'+SPLIT.upper()+'_NATIVE_METRICS_COMPLETE','utc':now(),'split':SPLIT,'arms':['G0','G1','G2'],'units_per_arm':expected,'native_scores':{a:m['aggregate_native'] for a,m in results.items()},'display_scores_x100':{a:m['aggregate_display_x100'] for a,m in results.items()},
  'diagnostics':{a:{k:m[k] for k in ['empty_predictions','empty_evidence_units','G2_invalid_selection_abstentions','output_token_cap_hits']} for a,m in results.items()},'metric_sha256':{a:sha(ART/(a+'.json')) for a in results},'generation_sha256':gen['generation_sha256'],'SPICE':'NOT_RUN','policy_adopted':False,'human_factuality_accuracy_established':False,'check_or_TEST_opened':False}
 if SPLIT=='dev':
  selected=max(['G0','G1','G2'],key=lambda a:results[a]['aggregate_native']['METEOR'])
  sample=json.loads((PREFLIGHT/'fixed20_dev_ids.json').read_text());assert len(sample)==20 and set(sample)<=ids
  save(ART/'fixed20_dev_support_packet.json',[{'unit_id':i,'references_evaluation_only':refs[i],'query_graph':queries[i],'outputs':{r['arm']:r['caption'] for r in rows if r['unit_id']==i},'raw_selection':next(r['raw_selection_output'] for r in rows if r['unit_id']==i and r['arm']=='G2'),'human_labels_created':0} for i in sample])
  result.update(selected_arm=selected,selection_rule='complete-corpus fresh-dev METEOR; exact ties G0 thenG1 thenG2',TRAIN_or_check_metrics_used_for_selection=False)
 save(ART/'complete.json',result)
 if SPLIT=='dev':
  with (PIPE/'selection_from_dev.json').open('x') as f:f.write(line({'utc':now(),'status':'CANDIDATE_LOCKED_FROM_DEV_FOR_FUTURE_SEPARATE_CHECK_ONLY','selected_arm':result['selected_arm'],'dev_metrics_completion_sha256':sha(ART/'complete.json'),'source_lock_sha256':sha(PIPE/'preregistration.json'),'policy_adopted':False,'automatic_check_or_TEST_launch':False})+'\n')
 print(line(result),flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:
  if ART.exists():save(ART/'failure.json',{'utc':now(),'status':'FAILED_NO_AUTOMATIC_RETRY','split':SPLIT,'reason':str(e)})
  traceback.print_exc();raise
