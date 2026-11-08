"""Exclusive finite grounding TRAIN/dev launch after CPU prompt/resource admission."""
from pathlib import Path
import json,os,shutil,subprocess,sys,time
from rsgformer.vg_grounding_train_dev_runtime_r1_20261103 import *
REPORT=ROOT/'reports/VG_GROUNDING_TRAIN_DEV_LAUNCH_20261103.json'
assert not PIPE.exists() and not REPORT.exists(),'exclusive_launch_no_duplicate'
cfg,manifest=verify_inputs()
for rel in manifest['source_sha256']:
 if rel.endswith('.py'):compile((ROOT/rel).read_text(),rel,'exec')
pre=json.loads((PREFLIGHT/'complete.json').read_text());assert pre['status']=='GROUNDING_NATIVE_PROMPTS_AND_G0_REUSE_PREFLIGHT_PASS' and pre['source_manifest_sha256']==sha(MANIFEST) and pre['G0_TRAIN_exact_reuse_matches']==800
for name,h in pre['file_sha256'].items():assert sha(PREFLIGHT/name)==h
free=shutil.disk_usage(ROOT).free;used=artifact_bytes(ROOT/'artifacts');gpu=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0])*1024**2
ram=int(next(s for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')).split()[1])*1024
assert free>=17*1024**3 and used+1024**3<=40*1024**3 and gpu>=38*1024**3 and ram>=14*1024**3,'resource_admission'
assert time.time()+19800<1794670321.2414355,'absolute_deadline_admission'
PIPE.mkdir();(PIPE/'source_manifest.json').write_bytes(MANIFEST.read_bytes());(PIPE/'config.json').write_bytes(CFG.read_bytes())
lock={'utc':now(),'source_sha256':manifest['source_sha256'],'source_manifest_sha256':sha(MANIFEST),'config_sha256':sha(CFG),'preflight_completion_sha256':sha(PREFLIGHT/'complete.json'),'parent_input_sha256':cfg['parent_files_sha256'],'dev_cache_input_sha256':cfg['dev_cache_files_sha256'],'prior_consumed_evidence':cfg['prior_consumed_evidence'],'maximum_new_Qwen_calls':1312,'G0_TRAIN_regeneration_allowed':False,'TRAIN_or_check_metric_policy_selection_allowed':False,'selection':'complete-corpus dev METEOR, exact ties G0/G1/G2','check_or_TEST_access_allowed':False,'automatic_retry':False,'automatic_policy_adoption':False,'admission':{'disk_free_bytes':free,'aggregate_artifacts_bytes':used,'GPU_free_bytes':gpu,'RAM_available_bytes':ram}}
save(PIPE/'preregistration.json',lock)
with (PIPE/'controller.stdout.log').open('x') as out,(PIPE/'controller.stderr.log').open('x') as err:
 child=subprocess.Popen([sys.executable,'-B',str(ROOT/'scripts/remote_vg_grounding_controller_r1_20261103.py')],cwd=ROOT,env=dict(os.environ),stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
record={'status':'LAUNCHED','utc':now(),'controller_pid':child.pid,'preregistration_sha256':sha(PIPE/'preregistration.json'),'source_manifest_sha256':sha(MANIFEST),'preflight_completion_sha256':lock['preflight_completion_sha256'],'check_or_TEST_access_allowed':False,'admission':lock['admission']}
save(REPORT,record);print(line(record),flush=True)
