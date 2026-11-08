"""Score complete VAL corpora once, choose common dev template, no TEST access."""
from pathlib import Path
from datetime import datetime,timezone
import gzip,hashlib,json,os,sys,traceback
ROOT=Path.home()/'rsg-former';BASE=ROOT/'artifacts/vg_caption_execution_v1_20261025'
partition=sys.argv[1];assert partition in ['dev','check']
ART=BASE/('caption_'+partition+'_v1');DEST=BASE/('caption_'+partition+'_metrics_v1')
assert (ART/'complete.json').exists() and not DEST.exists()
DEST.mkdir()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,indent=2));t.replace(p)
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:4]))
java=json.loads((ROOT/'reports/CAPTION_JAVA_PREPARATION.json').read_text())
j=Path(java['java_path_alias'].replace('${REMOTE_HOME}',str(Path.home())))
os.environ['PATH']=str(j.parent)+os.pathsep+os.environ['PATH']
os.environ['_JAVA_OPTIONS']='-Xmx2g -XX:ActiveProcessorCount=1 -XX:+UseSerialGC -XX:MaxMetaspaceSize=256m -XX:ReservedCodeCacheSize=128m'
from rsgformer.caption_metric_runtime import private_metric_runtime
from rsgformer.caption_metrics_v2 import compute_metrics
from rsgformer.vg_caption_prompt_v1 import ARMS
try:
 source={p:sha(ROOT/p) for p in ['scripts/remote_vg_caption_score_dev_check_v1.py','src/rsgformer/caption_metrics_v2.py','src/rsgformer/caption_metric_runtime.py']}
 save(DEST/'preregistration.json',{'partition':partition,'source_sha256':source,'generation_sha256':sha(ART/'generation_commit.jsonl'),
  'references_sha256':sha(BASE/'human_references_separate.json'),'same_original_references_all_arms':True,'TEST_access_allowed':False,
  'complete_corpus_CIDEr_IDF':True,'empty_outputs_retained':True,'include_SPICE':partition=='check'})
 with gzip.open(BASE/'region_units_no_color.jsonl.gz','rt') as f:units=[u for u in map(json.loads,f) if u['partition']=='caption_'+partition]
 assert len(units)==256
 ids={u['unit_id'] for u in units};all_refs=json.loads((BASE/'human_references_separate.json').read_text());refs={k:all_refs[k] for k in ids}
 rows=[json.loads(x) for x in (ART/'generation_commit.jsonl').read_text().splitlines()]
 expected_templates=['P1','P2'] if partition=='dev' else [json.loads((BASE/'caption_prompt_selection_v1.json').read_text())['selected_template']]
 assert len(rows)==len(expected_templates)*len(ARMS)*len(ids)
 results={}
 for template in expected_templates:
  for arm in ARMS:
   local=[r for r in rows if r['template']==template and r['arm']==arm]
   assert len(local)==len(ids) and {r['unit_id'] for r in local}==ids
   preds={r['unit_id']:r['caption'] for r in local}
   scratch=ROOT/('artifacts/caption_readiness/caption_'+partition+'_v1_'+template+'_'+arm);scratch.mkdir(exist_ok=False)
   with private_metric_runtime(ROOT,scratch) as runtime:metric=compute_metrics(refs,preds,runtime,include_spice=partition=='check')
   metric.update(arm=arm,template=template,images=64,provided_region_geometry=True,
     color_scope='VAL_candidate_only' if arm.startswith('C') else 'no_color',test_scored=False,
     hit_output_token_cap=sum(r['hit_token_cap'] for r in local),no_evidence_units=sum(r['no_evidence'] for r in local))
   save(DEST/(template+'_'+arm+'.json'),metric);results[template+'_'+arm]=metric['aggregate_native']
   save(DEST/'progress.json',{'completed_corpora':len(results),'planned_corpora':len(expected_templates)*6,'last_template':template,'last_arm':arm})
 selection=None
 if partition=='dev':
  means={template:sum(results[template+'_'+arm]['METEOR'] for arm in ARMS)/len(ARMS) for template in expected_templates}
  assert means['P1']!=means['P2'],'exact_METEOR_tie_requires_factuality_review_no_automatic_choice'
  choice=max(means,key=means.get)
  selection={'status':'COMMON_PROMPT_LOCKED_FROM_DEV_ONLY','utc':datetime.now(timezone.utc).isoformat(),'selected_template':choice,'mean_dev_METEOR_native':means,
   'dev_generation_sha256':sha(ART/'generation_commit.jsonl'),'bank_sha256':sha(ART/'train_demonstration_bank.json'),
   'no_TEST_access':True,'heldout_check_may_not_select_new_template':True,'colored_TEST_allowed':False}
  assert not (BASE/'caption_prompt_selection_v1.json').exists();save(BASE/'caption_prompt_selection_v1.json',selection)
 result={'status':'SIX_ARM_VAL_'+partition.upper()+'_METRICS_COMPLETE','utc':datetime.now(timezone.utc).isoformat(),'corpora':len(results),'units_per_corpus':len(ids),
  'native_scores':results,'selection':selection,'test_accessed':False,'published_comparison':'diagnostic manifest; not matched native VG caption benchmark',
  'metric_file_sha256':{p.name:sha(p) for p in DEST.glob('P*.json')}}
 save(DEST/'complete.json',result);save(ROOT/('reports/VG_CAPTION_'+partition.upper()+'_RESULT_20261025.json'),result);print(json.dumps(result),flush=True)
except Exception as e:
 save(DEST/'failure.json',{'error_type':type(e).__name__,'reason':str(e),'test_accessed':False});traceback.print_exc();raise
