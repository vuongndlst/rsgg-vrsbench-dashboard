from pathlib import Path
import json,subprocess,datetime,shutil
root=Path.home()/'rsg-former';a=root/'artifacts/vg_method_v2';out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'processes':{}}
pointer_path=root/'reports/VG_ACTIVE_EXPERIMENT.json';pointer=None
if pointer_path.exists():
 pointer=json.loads(pointer_path.read_text())
 assert pointer['experiment_id']=='vg_method_v2_speed_m2_v1'
 assert pointer['artifact_root']=='artifacts/vg_method_v2_speed_m2_v1'
 if pointer.get('execution_revision')=='vg_method_v2_time_extension_v1':
  assert pointer['pipeline_token']=='remote_vg_v2_time_extension_v1.py'
  assert pointer['pipeline_pid_path']=='artifacts/vg_method_v2_speed_m2_v1/time_extension_v1/pipeline.pid'
  assert pointer['progress_report']=='VG_V2_TIME_EXT_PROGRESS.json' and pointer['result_report']=='VG_V2_TIME_EXT_RESULT.json'
  assert pointer['config_path']=='configs/vg_method_v2_time_extension_v1.json'
  assert pointer['scientific_preregistration_sha256']=='d5866565e34555143af0f2f42c240841eaa8e1f2b0152c05c4ef60121561377c'
 else:
  assert not pointer.get('execution_revision'),'unknown_execution_revision'
  assert pointer['pipeline_token']=='remote_vg_v2_speed_pipeline.py'
  assert pointer['progress_report']=='VG_V2_SPEED_PROGRESS.json' and pointer['result_report']=='VG_V2_SPEED_RESULT.json'
  assert pointer['config_path']=='configs/vg_method_v2_speed_m2_v1.json'
 a=root/pointer['artifact_root'];out['active_experiment']=pointer
 out['active_config']=json.loads((root/pointer['config_path']).read_text())
for kind in ['prepare','pipeline']:
 p=root/pointer['pipeline_pid_path'] if pointer and kind=='pipeline' and pointer.get('pipeline_pid_path') else a/(kind+'.pid')
 if p.exists():
  pid=int(p.read_text());cmd=Path('/proc')/str(pid)/'cmdline'
  token=pointer['pipeline_token'].encode() if pointer and kind=='pipeline' else ('vg_v2_'+('prepare' if kind=='prepare' else 'pipeline')+'.py').encode()
  out['processes'][kind]={'pid':pid,'active':cmd.exists() and token in cmd.read_bytes()}
for n in ['VG_V2_PROGRESS.json','VG_V2_ASSETS.json','VG_V2_PROBE.json','VG_V2_RESULT.json','VG_V2_PREREGISTRATION.json','VG_REGION_REFERENCE_PREPARATION.json','VG_CAPTION_OVERLAP_AUDIT.json','VG_CANONICAL_ANNOTATION_PREPARATION.json','VG_CANONICAL_PARITY_AUDIT.json','OPENIMAGES_TEST_METADATA.json','CAPTION_CLEAN_IDENTITY_MERGE.json','OPENIMAGES_TEST_PIXEL_PILOT.json','FRESH_BACKBONE_INITIALIZATION_AUDIT.json']:
 p=root/'reports'/n
 if p.exists():
  value=json.loads(p.read_text())
  if n=='OPENIMAGES_TEST_PIXEL_PILOT.json':value.pop('records',None)
  out[n]=value
for n in ['VG_CROSS_SPLIT_PIXEL_AUDIT.json','CAPTION_INTERNAL_IDENTITY_MERGE.json',
          'COCO_PROTECTED_IDENTITY_PREPARATION.json','VG_PIXEL_FINGERPRINT_PROGRESS.json',
          'OPENIMAGES_NETWORK_READER_PROBE.json','QWEN_QUANTIZATION_PREFLIGHT.json',
          'CAPTION_STATISTICS_PROBE.json','EXTERNAL_PIXEL_FINGERPRINT_PROGRESS.json',
          'CAPTION_EXACT_PIXEL_COMPARISON.json','CAPTION_METRIC_RUNTIME_PROBE.json',
          'CAPTION_CORPUS_METRIC_PROBE.json','FRESH_DETECTOR_NATIVE_GPU_PROBE.json',
          'OI_STREAM_FINGERPRINT_PILOT.json','OI_STREAM_LEDGER_PROBE.json',
          'OI_FULL_STREAM_LEDGER_PROBE.json','OI_STREAM_PHASE_LAUNCH.json',
          'OI_STREAM_FINGERPRINT_PROGRESS.json','OI_STREAM_PHASE_RESULT.json',
          'CAPTION_EXACT_PIXEL_COMPARISON_V2.json','CAPTION_EXACT_PIXEL_COMPARISON_V3.json',
          'OI_EXHAUSTED_SOURCE_AUDIT.json','OI_FAILED_HEADER_PROBE_V2.json','OI_FAILED_HEADER_PROBE_V3.json',
          'OI_READER_V2_CPU_PROBE.json','OI_MPO_REPAIR_PILOT.json','VG_REGION_CAPTION_SCHEMA_PROBE.json']:
 p=root/'reports'/n
 if p.exists():out[n]=json.loads(p.read_text())
if pointer:
 out.pop('VG_V2_RESULT.json',None)
 p=root/'reports'/pointer['progress_report']
 out['VG_V2_PROGRESS.json']=json.loads(p.read_text()) if p.exists() else {'stage':'new_experiment_starting'}
 p=root/'reports'/pointer['result_report']
 if p.exists():out['VG_V2_RESULT.json']=json.loads(p.read_text())
# Ancillary audit must not count as VG training activity: otherwise a surviving
# OI worker could hide an unexpected VG process exit.
oi_phase=root/'artifacts/caption_readiness/oi_stream_phase_v1'
if (oi_phase/'pipeline.pid').exists():
 pid=int((oi_phase/'pipeline.pid').read_text());cmd=Path('/proc')/str(pid)/'cmdline'
 out['oi_stream_audit']={'pid':pid,'active':cmd.exists() and b'remote_oi_stream_pipeline.py' in cmd.read_bytes()}
 if (oi_phase/'status.json').exists():out['oi_stream_audit']['progress']=json.loads((oi_phase/'status.json').read_text())
out['downloads']=[{'name':p.name,'bytes':p.stat().st_size,'mtime':p.stat().st_mtime} for p in (root/'vendor/salience_sgg/logs').glob('*') if p.is_file()]
out['disk_free_bytes']=shutil.disk_usage(root).free
out['runs']=[]
for p in sorted(a.glob('*/report.json')):out['runs'].append(json.loads(p.read_text()))
out['logs']={}
for n in ['prepare.log','pipeline.log']:
 p=a/'time_extension_v1'/n if pointer and pointer.get('execution_revision') and n=='pipeline.log' else a/n
 if p.exists():
  with p.open('rb') as f:f.seek(max(0,p.stat().st_size-3000));out['logs'][n]=f.read().decode(errors='replace')
out['gpu']=subprocess.run(['nvidia-smi','--query-gpu=memory.free,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True,timeout=15).stdout
t=json.dumps(out,indent=2).replace(str(Path.home()),'${REMOTE_HOME}');print(t)
