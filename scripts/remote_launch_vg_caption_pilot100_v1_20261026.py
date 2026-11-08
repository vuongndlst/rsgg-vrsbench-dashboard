"""Exclusive human-authorized100image VAL caption pilot launch."""
from pathlib import Path
import json,os,shutil,subprocess,sys,time
ROOT=Path.home()/'rsg-former';PIPE=ROOT/'artifacts/vg_caption_pilot100_v1_20261026';REPORT=ROOT/'reports/VG_CAPTION_PILOT100_LAUNCH_20261026.json'
from rsgformer.vg_color_execution_common_v1 import sha,save,line,now,artifact_bytes
assert not PIPE.exists() and not REPORT.exists(),'no_duplicate_launch'
manifest=json.loads((ROOT/'configs/vg_caption_pilot100_source_manifest_v1_20261026.json').read_text())
for p,h in manifest['source_sha256'].items():assert sha(ROOT/p)==h,('local_remote_source_sync',p)
for p in manifest['source_sha256']:
 if p.endswith('.py'):compile((ROOT/p).read_text(),p,'exec')
assert json.loads((ROOT/'artifacts/vg_color_auto_full_pipeline_v1_20261025/complete.json').read_text())['status']=='COMPLETE_AUTOMATED_COLOR_TEST_NOT_HUMAN_PRECISION'
assert json.loads((ROOT/'reports/QWEN_CAPTION_GPU_PILOT_V2_20261025.json').read_text())['status']=='QWEN_BF16_SYNTHETIC_GPU_PILOT_PASS'
assert json.loads((ROOT/'reports/VG_CAPTION_CONTRACTS_V1_20261025.json').read_text())['status']=='PROMPT_AND_NATIVE_METRIC_CONTRACT_PASS'
used=artifact_bytes(ROOT/'artifacts');free=shutil.disk_usage(ROOT).free
gpu=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0])*1024**2
ram=int(next(v for v in Path('/proc/meminfo').read_text().splitlines() if v.startswith('MemAvailable:')).split()[1])*1024
assert free>=17*1024**3 and used+1024**3<=40*1024**3 and gpu>=38*1024**3 and ram>=14*1024**3
assert time.time()+13500<1794670321.2414355
PIPE.mkdir();lock={'utc':now(),'authorization':'26Oct user requested pilot captions/evaluation100images with multiple configurations; prior Qwen36/32/12GiB CPU4 approved conditionally',
 'source_sha256':manifest['source_sha256'],'config_sha256':sha(ROOT/'configs/vg_caption_pilot100_v1_20261026.json'),
 'base_cache_completion_sha256':sha(ROOT/'artifacts/vg_caption_execution_v1_20261025/base_cache_complete.json'),
 'admission':{'disk_free_bytes':free,'aggregate_artifacts_bytes':used,'GPU_free_bytes':gpu,'host_RAM_available_bytes':ram},
 'test_access_allowed':False,'candidate_color_only':True,'automatic_retry':False,'automatic_prompt_or_TEST_selection':False}
save(PIPE/'preregistration.json',lock)
with (PIPE/'controller.stdout.log').open('x') as out,(PIPE/'controller.stderr.log').open('x') as err:
 child=subprocess.Popen([sys.executable,'-B',str(ROOT/'scripts/remote_vg_caption_pilot100_controller_v1_20261026.py')],cwd=ROOT,env=dict(os.environ),stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
record={'status':'LAUNCHED','utc':now(),'controller_pid':child.pid,'preregistration_sha256':sha(PIPE/'preregistration.json'),'admission':lock['admission'],'TEST_access_allowed':False}
save(REPORT,record);print(line(record))
