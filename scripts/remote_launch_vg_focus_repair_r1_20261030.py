"""Exclusive audited continuation after verified failure, resource guards unchanged."""
from pathlib import Path
import json,os,shutil,subprocess,sys,time
ROOT=Path.home()/'rsg-former';PARENT=ROOT/'artifacts/vg_focus_repair_v1_20261028';PIPE=ROOT/'artifacts/vg_focus_repair_r1_20261030';REPORT=ROOT/'reports/VG_FOCUS_REPAIR_R1_LAUNCH_20261030.json'
from rsgformer.vg_color_execution_common_v1 import sha,save,line,now,artifact_bytes
assert not PIPE.exists() and not REPORT.exists(),'no_duplicate_launch'
manifest=json.loads((ROOT/'configs/vg_focus_repair_r1_source_manifest_20261030.json').read_text());config=json.loads((ROOT/'configs/vg_focus_repair_r1_20261030.json').read_text())
for p,h in manifest['source_sha256'].items():assert sha(ROOT/p)==h,('source_integrity',p)
for p in manifest['source_sha256']:
 if p.endswith('.py'):compile((ROOT/p).read_text(),p,'exec')
for p,h in config['parent_file_sha256'].items():assert sha(PARENT/p)==h,('parent_evidence_drift',p)
assert not (PARENT/'check_access.json').exists() and not (PARENT/'check_generation').exists() and not (PARENT/'check_metrics').exists()
assert sha(PARENT/'preregistration.json')==config['parent_preregistration_sha256']
failure=json.loads((PARENT/'failure.json').read_text());assert failure['reason']=='GPU_admission' and not failure['check_access_opened']
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:args=(p/'cmdline').read_bytes().split(b'\x00')
 except OSError:continue
 assert str(ROOT/'scripts/remote_vg_focus_repair_controller_v1_20261028.py').encode() not in args,'parent_controller_active'
selection=json.loads((PARENT/'selection.json').read_text());assert selection['selected_policy']=='L2' and selection['TRAIN_metrics_completion_sha256']==sha(PARENT/'train_metrics/complete.json')
free=shutil.disk_usage(ROOT).free;used=artifact_bytes(ROOT/'artifacts');gpu=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0])*1024**2
ram=int(next(v for v in Path('/proc/meminfo').read_text().splitlines() if v.startswith('MemAvailable:')).split()[1])*1024
assert free>=17*1024**3 and used+1024**3<=40*1024**3 and gpu>=38*1024**3 and ram>=14*1024**3,'unchanged_resource_admission'
assert time.time()+9900<1794670321.2414355
PIPE.mkdir();links={n:'artifacts/vg_focus_repair_v1_20261028/'+n for n in ['prepare','train_generation','train_metrics']}
for n,target in links.items():(PIPE/n).symlink_to(ROOT/target,target_is_directory=True)
with (PIPE/'selection.json').open('xb') as f:f.write((PARENT/'selection.json').read_bytes())
lock={'utc':now(),'source_sha256':manifest['source_sha256'],'config_sha256':sha(ROOT/'configs/vg_focus_repair_r1_20261030.json'),'parent_preregistration_sha256':config['parent_preregistration_sha256'],'parent_file_sha256':config['parent_file_sha256'],'parent_selection_sha256':sha(PARENT/'selection.json'),'reused_readonly_links':links,'prior_consumed_outcomes':config['prior_consumed_outcomes'],'check_previously_opened':False,'TRAIN_regeneration_allowed':False,'new_check_max_outputs':512,'selection_changed':False,'check_policy_selection_allowed':False,'TEST_access_allowed':False,'automatic_retry':False,'automatic_policy_adoption':False,'admission':{'disk_free_bytes':free,'aggregate_artifacts_bytes':used,'GPU_free_bytes':gpu,'host_RAM_available_bytes':ram}}
save(PIPE/'preregistration.json',lock)
with (PIPE/'controller.stdout.log').open('x') as out,(PIPE/'controller.stderr.log').open('x') as err:
 child=subprocess.Popen([sys.executable,'-B',str(ROOT/'scripts/remote_vg_focus_repair_controller_r1_20261030.py')],cwd=ROOT,env=dict(os.environ),stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
record={'status':'LAUNCHED','utc':now(),'controller_pid':child.pid,'preregistration_sha256':sha(PIPE/'preregistration.json'),'parent_preregistration_sha256':config['parent_preregistration_sha256'],'TRAIN_reused_not_regenerated':True,'TEST_access_allowed':False,'admission':lock['admission']}
save(REPORT,record);print(line(record))
