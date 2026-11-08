"""Frozen automatic attribute diagnostic, permitted only after full cache commit."""
from pathlib import Path
import json,os,signal,sqlite3,traceback
ROOT=Path.home()/'rsg-former';ART=ROOT/'artifacts/vg_color_auto_test_diagnostic_v1_20261025';CACHE=ROOT/'artifacts/vg_color_full_test_auto_v1_20261025';PIPE=ROOT/'artifacts/vg_color_auto_full_pipeline_v1_20261025'
os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:4]));signal.alarm(3600)
os.environ.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
from rsgformer.vg_color_execution_common_v1 import Guard,sha,save,line,now
from rsgformer.vg_color_attribute_diagnostic_v1 import evaluate

def main():
 assert not ART.exists(),'no_metric_retry_or_partial_resume'
 guard=Guard(ROOT,ART,3600,.25);guard.check(False)
 lock=json.loads((PIPE/'preregistration.json').read_text())
 for p,h in lock['source_sha256'].items():assert sha(ROOT/p)==h
 complete=json.loads((CACHE/'complete.json').read_text());assert complete['counts']['images']==26446
 for p,h in complete['output_sha256'].items():assert sha(CACHE/p)==h
 ART.mkdir();save(ART/'preregistration.json',{'utc':now(),'pipeline_lock_sha256':sha(PIPE/'preregistration.json'),'full_cache_completion_sha256':sha(CACHE/'complete.json'),
  'diagnostic_source_sha256':sha(ROOT/'src/rsgformer/vg_color_attribute_diagnostic_v1.py'),'same_algorithm_as_VAL':True,'GT_never_given_to_extractor':True,'human_precision_gate_passed':False})
 with sqlite3.connect((ROOT/'artifacts/caption_readiness/vg_pixel_fingerprints_v1.sqlite').as_uri()+'?mode=ro',uri=True) as db:expected={int(r[0]) for r in db.execute("SELECT vg_id FROM pixels WHERE split='test'")}
 assert len(expected)==26446
 def progress(images,counts):guard.check(False);save(ART/'progress.json',{'utc':now(),'images_with_nodes_evaluated':images,'counts':counts})
 result=evaluate(ROOT,CACHE/'candidate_records.jsonl.gz',ART/'diagnostic',expected,expected_split='test',progress=progress)
 assert result['counts']['selected_nodes']==complete['counts']['requested_nodes'];guard.check(False)
 result.update(utc=now(),full_color_cache_completion_sha256=sha(CACHE/'complete.json'),TEST_used_for_tuning=False)
 save(ART/'complete.json',result);save(ROOT/'reports/VG_COLOR_AUTO_TEST_DIAGNOSTIC_V1_20261025.json',result);print(line(result),flush=True)
if __name__=='__main__':
 try:main()
 except Exception as error:
  if ART.exists():save(ART/'failure.json',{'utc':now(),'reason':str(error),'error_type':type(error).__name__,'automatic_retry_allowed':False})
  traceback.print_exc();raise
