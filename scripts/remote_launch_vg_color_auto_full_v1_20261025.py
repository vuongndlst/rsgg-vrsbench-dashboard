"""Exclusive launch after measured admission; locks every execution source before TEST."""
from pathlib import Path
import json,os,resource,shutil,subprocess,sys,time
ROOT=Path.home()/'rsg-former';ART=ROOT/'artifacts/vg_color_auto_full_pipeline_v1_20261025';REPORT=ROOT/'reports/VG_COLOR_AUTO_FULL_LAUNCH_V1_20261025.json'
from rsgformer.vg_color_execution_common_v1 import sha,save,line,now,artifact_bytes
assert not ART.exists() and not REPORT.exists(),'exclusive_launch_no_duplicate'
assert time.time()+90000<1794670321.2414355,'remaining_authorized_deadline'
free=shutil.disk_usage(ROOT).free;used=artifact_bytes(ROOT/'artifacts')
assert free>=23*1024**3 and used+7*1024**3<=40*1024**3,'worst_case_storage_admission'
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
assert int(gpu.splitlines()[0].split(',')[0])>=8*1024,'GPU8GiB_admission'
mem={r.split(':')[0]:int(r.split()[1])*1024 for r in Path('/proc/meminfo').read_text().splitlines() if ':' in r}
assert mem['MemAvailable']>=6*1024**3,'RSS4GiB_plus_host_headroom'
assert len(os.sched_getaffinity(0))>=4
val=ROOT/'artifacts/vg_color_auto_val_diagnostic_v1_20261025/complete.json';v=json.loads(val.read_text());assert v['counts']['selected_nodes']==3083
assert v['source_sha256']==sha(ROOT/'src/rsgformer/vg_color_attribute_diagnostic_v1.py')
speed=json.loads((ROOT/'artifacts/vg_color_speed_benchmark_v1_20261025/complete.json').read_text());assert speed
assert speed['parity']['safe_cache']['adoption_ready_engineering_only'] and speed['historical_TRAIN_reference_reproduced']
scripts=['remote_vg_color_fast_val_replay_v1_20261025.py','remote_vg_color_full_test_auto_v1_20261025.py','remote_vg_color_auto_test_diagnostic_v1_20261025.py','remote_vg_color_auto_full_controller_v1_20261025.py','remote_launch_vg_color_auto_full_v1_20261025.py']
modules=['vg_color_execution_common_v1.py','vg_color_full_test_policy_v1.py','vg_color_fast_engine_v1.py','vg_color_attribute_diagnostic_v1.py','vg_object_color_rules_v3.py','vg_object_color_v2.py','vg_color_binding_native_v1.py','sam_binary_mask_adapter_v1.py','numpy_json_adapter_v1.py','vg_caption_graph_v1.py']
sources=['scripts/'+p for p in scripts]+['src/rsgformer/'+p for p in modules]+['configs/vg_object_color_rules_v3_20261025.json','configs/vg_color_object_binding_native_v1_20261025.json','third_party/sam2_color_audit_20261025/RSGG_PROVENANCE.json']
manifest=json.loads((ROOT/'configs/vg_color_auto_full_source_manifest_v1_20261025.json').read_text())
assert set(manifest['source_sha256'])==set(p for p in sources if not p.startswith('third_party/'))|{'configs/vg_color_automated_full_test_v1_20261025.json'}
for p,h in manifest['source_sha256'].items():assert sha(ROOT/p)==h,('local_remote_sync_integrity',p)
for p in sources:
 if p.endswith('.py'):compile((ROOT/p).read_text(),p,'exec')
lock={'utc':now(),'phase':'vg_color_auto_full_pipeline_v1_20261025','authorization':'25Oct user: automated evaluation with explicit limits, then full colorTEST',
 'source_sha256':{p:sha(ROOT/p) for p in sources},'config_sha256':sha(ROOT/'configs/vg_color_automated_full_test_v1_20261025.json'),
 'VAL_auto_diagnostic_completion_sha256':sha(val),'human_precision_gate_passed':False,'independent_human_labels':0,
 'quality_statistic_is_descriptive_no_posthoc_threshold':True,'no_scientific_tuning_after_VAL_check':True,
 'admission':{'free_disk_bytes':free,'aggregate_artifacts_bytes':used,'GPU_free_used_utilization':gpu,'available_host_RAM_bytes':mem['MemAvailable'],'CPU_affinity_available':len(os.sched_getaffinity(0))},
 'stages':['all3083VAL_engineering_parity','first_full26446_color_cache','postcache_frozen_attribute_diagnostic'],'no_automatic_retry':True,
 'caps':json.loads((ROOT/'configs/vg_color_automated_full_test_v1_20261025.json').read_text())['limits']}
ART.mkdir();save(ART/'preregistration.json',lock)
with (ART/'controller.stdout.log').open('x') as out,(ART/'controller.stderr.log').open('x') as err:
 child=subprocess.Popen([sys.executable,'-B',str(ROOT/'scripts/remote_vg_color_auto_full_controller_v1_20261025.py')],cwd=ROOT,env=dict(os.environ),stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
record={'utc':now(),'status':'LAUNCHED','controller_pid':child.pid,'preregistration_sha256':sha(ART/'preregistration.json'),'admission':lock['admission'],'human_precision_gate_passed':False}
save(REPORT,record);print(line(record))
