"""TRAIN/dev only controller using immutable owned-tree watchdog, no check launch."""
from pathlib import Path
import importlib.util,json,os,traceback,zipfile
from rsgformer.vg_grounding_train_dev_runtime_v1_20261103 import *
REPORT=ROOT/'reports/VG_GROUNDING_TRAIN_DEV_CONTROLLER_20261103.json'
spec=importlib.util.spec_from_file_location('frozen_owned_watchdog',ROOT/'scripts/remote_vg_focus_repair_controller_v1_20261028.py');watch=importlib.util.module_from_spec(spec);spec.loader.exec_module(watch);watch.PIPE=PIPE;watch.REPORT=REPORT
base_artifact_bytes=watch.artifact_bytes

def owned_size(path):
 if Path(path)!=PIPE:return base_artifact_bytes(path)
 return base_artifact_bytes(PIPE)+base_artifact_bytes(PREFLIGHT)+sum(base_artifact_bytes(p) for p in (ROOT/'artifacts/caption_readiness').glob('grounding_train_dev_v1_20261103_*') if p.is_dir())
watch.artifact_bytes=owned_size

def main():
 assert not (PIPE/'started.json').exists();save(PIPE/'started.json',{'utc':now(),'pid':os.getpid()});cfg,manifest=verify_inputs();lock=json.loads((PIPE/'preregistration.json').read_text())
 assert sha(PREFLIGHT/'complete.json')==lock['preflight_completion_sha256']
 outcomes=[]
 for split in ['train','dev']:
  os.environ['VG_GROUNDING_SPLIT']=split
  outcomes.append(watch.run_stage(split+'_generation','remote_vg_grounding_generate_v1_20261103.py',36,12,7200,lock))
  outcomes.append(watch.run_stage(split+'_metrics','remote_vg_grounding_score_v1_20261103.py',0,4,2700,lock))
 train=json.loads((PIPE/'train_generation/complete.json').read_text());dev=json.loads((PIPE/'dev_generation/complete.json').read_text())
 assert train['records']==2400 and dev['records']==768 and train['new_model_calls']+dev['new_model_calls']<=1312
 assert train['G0_TRAIN_reused']==800
 assert not (PIPE/'check_access.json').exists() and not (PIPE/'check_generation').exists() and not (PIPE/'check_metrics').exists()
 selection=json.loads((PIPE/'selection_from_dev.json').read_text())
 result={'status':'GROUNDING_FINITE_TRAIN_DEV_COMPLETE_NO_HELDOUT_OR_TEST','utc':now(),'controller_pid':os.getpid(),'preregistration_sha256':sha(PIPE/'preregistration.json'),'source_manifest_sha256':sha(MANIFEST),'outcomes':outcomes,'selected_arm_from_dev':selection['selected_arm'],'selection_sha256':sha(PIPE/'selection_from_dev.json'),'TRAIN_generation_records':2400,'dev_generation_records':768,'new_model_calls':train['new_model_calls']+dev['new_model_calls'],'G0_TRAIN_reused':800,'TRAIN_metric_complete_sha256':sha(PIPE/'train_metrics/complete.json'),'dev_metric_complete_sha256':sha(PIPE/'dev_metrics/complete.json'),'policy_adopted':False,'human_factuality_accuracy_established':False,'check_or_TEST_opened':False,'automatic_check_or_TEST_launch':False}
 save(PIPE/'stages_complete.json',result)
 files=[p for p in PIPE.rglob('*') if p.is_file() and p.suffix!='.log' and p.name!='status.json'];pre_files=[p for p in PREFLIGHT.iterdir() if p.is_file()]
 assert sum(p.stat().st_size for p in files+pre_files)<64*1024**2
 archive=PIPE/'grounding_train_dev_results.zip'
 with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
  for p in files:z.write(p,p.relative_to(PIPE).as_posix())
  for p in pre_files:z.write(p,'preflight/'+p.name)
 result.update(archive=archive.relative_to(ROOT).as_posix(),archive_sha256=sha(archive),archive_bytes=archive.stat().st_size)
 assert owned_size(PIPE)<1024**3
 save(PIPE/'complete.json',result);save(PIPE/'status.json',result);save(REPORT,result);print(line(result),flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:
  s=json.loads((PIPE/'status.json').read_text()) if (PIPE/'status.json').exists() else {}
  s.update(status='FAILED_NO_AUTOMATIC_RETRY',utc=now(),reason=str(e),check_or_TEST_opened=False)
  save(PIPE/'failure.json',s);save(PIPE/'status.json',s);save(REPORT,s);traceback.print_exc();raise
