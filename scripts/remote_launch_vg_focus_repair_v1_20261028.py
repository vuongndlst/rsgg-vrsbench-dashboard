"""Exclusive launch after source/contracts and measured Qwen resource admission."""
from pathlib import Path
import json,os,shutil,subprocess,sys,time
ROOT=Path.home()/'rsg-former';PIPE=ROOT/'artifacts/vg_focus_repair_v1_20261028';REPORT=ROOT/'reports/VG_FOCUS_REPAIR_LAUNCH_20261028.json'
from rsgformer.vg_color_execution_common_v1 import sha,save,line,now,artifact_bytes
from rsgformer.vg_region_focus_repair_v1 import synthetic_contract
assert not PIPE.exists() and not REPORT.exists(),'no_duplicate_launch'
manifest=json.loads((ROOT/'configs/vg_focus_repair_source_manifest_v1_20261028.json').read_text())
for p,h in manifest['source_sha256'].items():assert sha(ROOT/p)==h,('local_remote_source_integrity',p)
for p in manifest['source_sha256']:
 if p.endswith('.py'):compile((ROOT/p).read_text(),p,'exec')
contract=synthetic_contract()
assert json.loads((ROOT/'reports/QWEN_CAPTION_GPU_PILOT_V2_20261025.json').read_text())['status']=='QWEN_BF16_SYNTHETIC_GPU_PILOT_PASS'
assert json.loads((ROOT/'reports/VG_CAPTION_CONTRACTS_V1_20261025.json').read_text())['status']=='PROMPT_AND_NATIVE_METRIC_CONTRACT_PASS'
free=shutil.disk_usage(ROOT).free;used=artifact_bytes(ROOT/'artifacts');gpu=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0])*1024**2
ram=int(next(v for v in Path('/proc/meminfo').read_text().splitlines() if v.startswith('MemAvailable:')).split()[1])*1024
assert free>=17*1024**3 and used+1024**3<=40*1024**3 and gpu>=38*1024**3 and ram>=14*1024**3
assert time.time()+20400<1794670321.2414355
PIPE.mkdir();lock={'utc':now(),'authorization':'continued finite TRAIN/heldout-check focus repair within authorized research scope after27Oct audit; caps unchanged',
 'source_sha256':manifest['source_sha256'],'config_sha256':sha(ROOT/'configs/vg_focus_repair_v1_20261028.json'),
 'synthetic_contract':contract,'admission':{'disk_free_bytes':free,'aggregate_artifacts_bytes':used,'GPU_free_bytes':gpu,'host_RAM_available_bytes':ram},
 'selection':'complete-corpus TRAIN METEOR, exact ties L0thenL1thenL2','check_policy_selection_allowed':False,'TEST_access_allowed':False,'automatic_retry':False,'automatic_policy_adoption':False}
save(PIPE/'preregistration.json',lock)
with (PIPE/'controller.stdout.log').open('x') as out,(PIPE/'controller.stderr.log').open('x') as err:
 child=subprocess.Popen([sys.executable,'-B',str(ROOT/'scripts/remote_vg_focus_repair_controller_v1_20261028.py')],cwd=ROOT,env=dict(os.environ),stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
record={'status':'LAUNCHED','utc':now(),'controller_pid':child.pid,'preregistration_sha256':sha(PIPE/'preregistration.json'),'admission':lock['admission'],'TEST_access_allowed':False}
save(REPORT,record);print(line(record))
