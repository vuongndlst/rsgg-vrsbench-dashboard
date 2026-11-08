"""Audited continuation: immutable TRAIN reuse, one unopened check, original watchdog."""
from pathlib import Path
import importlib.util,json,os,traceback,zipfile
ROOT=Path.home()/'rsg-former';PIPE=ROOT/'artifacts/vg_focus_repair_r1_20261030';REPORT=ROOT/'reports/VG_FOCUS_REPAIR_R1_CONTROLLER_20261030.json'
from rsgformer.vg_color_execution_common_v1 import sha,save,line,now
spec=importlib.util.spec_from_file_location('frozen_focus_parent_watchdog',ROOT/'scripts/remote_vg_focus_repair_controller_v1_20261028.py');watch=importlib.util.module_from_spec(spec);spec.loader.exec_module(watch);watch.PIPE=PIPE;watch.REPORT=REPORT

def main():
 assert not (PIPE/'started.json').exists();save(PIPE/'started.json',{'utc':now(),'pid':os.getpid()})
 lock=json.loads((PIPE/'preregistration.json').read_text());watch.source_check(lock)
 for name,target in lock['reused_readonly_links'].items():assert (PIPE/name).is_symlink() and (PIPE/name).resolve()==ROOT/target
 assert sha(PIPE/'selection.json')==lock['parent_selection_sha256']
 outcomes=[];os.environ['VG_FOCUS_STAGE']='check'
 outcomes.append(watch.run_stage('check_generation','remote_vg_focus_repair_generate_r1_20261030.py',36,12,7200,lock))
 outcomes.append(watch.run_stage('check_metrics','remote_vg_focus_repair_score_r1_20261030.py',0,4,2700,lock))
 check=json.loads((PIPE/'check_metrics/complete.json').read_text());selection=json.loads((PIPE/'selection.json').read_text())
 assert check['selected_policy']==selection['selected_policy']=='L2'
 result={'status':'FOCUS_REPAIR_R1_REUSED_TRAIN_AND_FROZEN_CHECK_COMPLETE','utc':now(),'controller_pid':os.getpid(),'outcomes':outcomes,'selected_policy':'L2','preregistration_sha256':sha(PIPE/'preregistration.json'),'parent_preregistration_sha256':lock['parent_preregistration_sha256'],'TRAIN_reused_not_regenerated':True,'TRAIN_completion_sha256':sha(PIPE/'train_metrics/complete.json'),'check_completion_sha256':sha(PIPE/'check_metrics/complete.json'),'selection_sha256':sha(PIPE/'selection.json'),'policy_adopted':False,'TEST_accessed':False,'human_factuality_review_complete':False,'automatic_TEST_launch':False}
 save(PIPE/'stages_complete.json',result)
 files=[p for p in PIPE.rglob('*') if p.is_file() and not p.is_symlink() and p.suffix!='.log' and p.name!='status.json'];assert sum(p.stat().st_size for p in files)<64*1024**2
 archive=PIPE/'focus_repair_r1_results.zip'
 with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
  for p in files:z.write(p,p.relative_to(PIPE).as_posix())
 result.update(archive=str(archive.relative_to(ROOT)),archive_sha256=sha(archive),archive_bytes=archive.stat().st_size)
 save(PIPE/'complete.json',result);save(PIPE/'status.json',result);save(REPORT,result);print(line(result),flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:
  state=json.loads((PIPE/'status.json').read_text()) if (PIPE/'status.json').exists() else {}
  state.update(status='FAILED_NO_AUTOMATIC_RETRY',utc=now(),reason=str(e),TEST_accessed=False,check_access_opened=(PIPE/'check_access.json').exists())
  save(PIPE/'failure.json',state);save(PIPE/'status.json',state);save(REPORT,state);traceback.print_exc();raise
