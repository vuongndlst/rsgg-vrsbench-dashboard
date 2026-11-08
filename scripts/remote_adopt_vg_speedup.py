"""Once-only audited transition to new matched microbatch2 experiment."""
from pathlib import Path
from datetime import datetime,timezone
import ast,hashlib,json,os,shutil,subprocess,sys,time,traceback
root=Path.home()/'rsg-former';old=root/'artifacts/vg_method_v2'
new=root/'artifacts/vg_method_v2_speed_m2_v1';audit=new/'superseded_original'
cfg=json.loads((root/'configs/vg_method_v2_speed_m2_v1.json').read_text())
benchmark=root/'reports/VG_SPEEDUP_BENCHMARK.json';controller=root/'reports/VG_SPEEDUP_CONTROLLER.json'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for part in iter(lambda:f.read(8*1024*1024),b''):h.update(part)
 return h.hexdigest()
def atomic(p,obj):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(obj,indent=2));t.replace(p)
def alive(pid,token):
 p=Path('/proc')/str(pid)/'cmdline'
 return p.exists() and token in p.read_bytes()
def preserve(rel):
 p=root/rel
 if not p.exists():return None
 target=audit/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
 return {'path':rel,'sha256':sha(target),'bytes':target.stat().st_size}
assert not new.exists(),'no_duplicate_new_experiment'
assert not (old/'STOP').exists(),'existing_stop_no_override'
assert sha(benchmark)=='bec2a2cfbfd39e5edf52c7bfc7806385e061d1f4a5777a22e95b52b32b7ef4e3'
assert sha(controller)=='84d6e490df568ac2a346611c5647cdfd7dad283c386fb77dedd346e12ca4863c'
b=json.loads(benchmark.read_text());c=json.loads(controller.read_text())
assert b['status']=='BENCHMARK_COMPLETE' and b['adoption_ready'] and b['selected_case']=='packed_fp32_m2'
assert b['source_lock_still_verified'] and not b['benchmark_test_accessed'] and not b['validation_scored']
assert all(b['parity'][v]['2']['status']=='PASS' and b['cases'][v]['packed_fp32_m2']['status']=='PASS' for v in cfg['screening']['variants'])
assert c['status']=='BENCHMARK_FINISHED_ORIGINAL_V2_RESUMED'
assert cfg['original_wall_deadline_unix']==c['original_wall_deadline_unix']
assert time.time()<cfg['original_wall_deadline_unix']
assert shutil.disk_usage(root).free>16*1024**3+8*1024**3,'disk_checkpoint_reserve'
pid=int((old/'pipeline.pid').read_text());assert pid==c['resumed_pid'] and alive(pid,b'remote_vg_v2_pipeline.py')
original_lock=json.loads((root/'reports/VG_V2_PREREGISTRATION.json').read_text())
sources=dict(b['protocol']['source_hashes'])
for name in ['scripts/remote_vg_v2_speed_pipeline.py','scripts/remote_adopt_vg_speedup.py',
             'src/rsgformer/vg_speedup_schedule.py','src/rsgformer/data.py','src/rsgformer/geometry.py',
             'configs/vg_method_v2_speed_m2_v1.json','vendor/salience_sgg/configs/SalienceSGG_vg.yaml']:
 sources[name]=sha(root/name)
for name,h in sources.items():assert sha(root/name)==h,name
assert original_lock['source_hashes']=={name:sources[name] for name in original_lock['source_hashes']}
for name in ['scripts/remote_vg_v2_speed_pipeline.py','src/rsgformer/vg_speedup_schedule.py']:
 ast.parse((root/name).read_text())
sys.path.insert(0,str(root/'src'))
from rsgformer.vg_speedup_schedule import grouped,step_boundary
contracts=[]
for n in [0,1,15,16,17,31,32,33,64,57720]:
 groups=list(grouped(range(n),2));assert [x for g in groups for x in g]==list(range(n))
 images=0;updates=[]
 for g in groups:
  images+=len(g)
  if step_boundary(images,16):updates.append(images)
 if images%16:updates.append(images)
 assert updates==list(range(16,n+1,16))+([n] if n%16 else []),'effective_batch_or_tail_changed'
 contracts.append(n)
manifest_path=old/'features/manifest.json';manifest=json.loads(manifest_path.read_text())
assert manifest['status']=='complete' and manifest['cache_dtype']=='float32'
assert manifest['preregistration_source_hashes']==original_lock['source_hashes']
assert sha(manifest_path)==b['protocol']['manifest_sha256']
for split in ['train','val']:
 for entry in manifest['splits'][split]:
  assert (old/'features'/entry['path']).stat().st_size==entry['bytes'],'cache_shard_stat_changed'
native=json.loads((old/'screening_native_1729/report.json').read_text())
assert native['status']=='complete' and sha(root/native['checkpoint'])==native['checkpoint_sha256']
baseline={name:sha(root/name) for name in ['artifacts/vg_method_v2/frozen_native_validation.json','artifacts/vg_method_v2/frozen_native_validation_records.json']}
# Native measured epoch times, rather than the short benchmark, bound admission.
screen_required=max(e['seconds'] for e in native['history'])*24*1.25
assert screen_required<cfg['original_wall_deadline_unix']-time.time(),'screening_admission_outside_original_cap'
prereg={'utc':datetime.now(timezone.utc).isoformat(),'config':cfg,'source_hashes':sources,
        'benchmark_evidence_sha256':sha(benchmark),'controller_evidence_sha256':sha(controller),
        'original_preregistration_sha256':sha(root/'reports/VG_V2_PREREGISTRATION.json'),
        'cache_manifest_sha256':sha(manifest_path),'split_hashes':original_lock['split_hashes'],
        'assets':original_lock['assets'],'frozen_baseline_source_hashes':baseline,
        'noncache_artifact_reserve_bytes':8*1024**3,
        'screening_required_seconds_conservative':screen_required,
        'all_four_arms_restart_from_original_initialization':True,'old_native_not_matched_control':True,
        'batch_schedule_contracts_pass':contracts,'test_scored':False,
        'authorization':'Human11Oct2026 approved separate speed work; heartbeat explicitly authorizes new matched experiment if adoption_ready'}
new.mkdir();audit.mkdir();atomic(new/'preregistration.json',prereg);atomic(root/'reports/VG_V2_SPEED_PREREGISTRATION.json',prereg)
stop_token=sha(new/'preregistration.json');created_stop=False
out={'experiment_id':cfg['experiment_id'],'status':'ADOPTION_PREPARED','preregistration_sha256':stop_token,
     'original_wall_deadline_unix':cfg['original_wall_deadline_unix'],'old_pid':pid,
     'original_sources_modified':False,'test_scored':False}
try:
 before=[preserve(p) for p in ['reports/VG_V2_PREREGISTRATION.json','reports/VG_V2_PROGRESS.json','artifacts/vg_method_v2/screening_native_1729/report.json']]
 assert alive(pid,b'remote_vg_v2_pipeline.py') and not (old/'STOP').exists()
 with (old/'STOP').open('x') as f:f.write('superseded:'+stop_token)
 created_stop=True;deadline=time.monotonic()+90
 while alive(pid,b'remote_vg_v2_pipeline.py'):
  assert time.monotonic()<deadline,'cooperative_stop_timeout_no_force_kill'
  time.sleep(.5)
 result=json.loads((root/'reports/VG_V2_RESULT.json').read_text())
 assert result.get('reason')=='user_stop_flag' and not result.get('test_scored'),'stop_outcome_requires_review'
 after=[preserve(p) for p in ['reports/VG_V2_RESULT.json','reports/VG_V2_PROGRESS.json','artifacts/vg_method_v2/pipeline.log']]
 runs=[]
 for p in old.glob('screening_*'):
  if p.is_dir():runs.append({'folder':p.name,'files':[{'name':q.name,'bytes':q.stat().st_size,'sha256':sha(q)} for q in p.iterdir() if q.is_file()]})
 assert all(sha(root/name)==h for name,h in sources.items()),'source_changed_before_launch'
 assert (old/'STOP').read_text()=='superseded:'+stop_token,'stop_ownership_changed'
 # Old partial/completed artifacts stay intact. STOP remains to prevent blind old resume.
 supersession={'utc':datetime.now(timezone.utc).isoformat(),'status':'SUPERSEDED_BY_NEW_MATCHED_MICROBATCH_PROTOCOL',
               'old_pid':pid,'new_experiment':cfg['experiment_id'],'original_native_checkpoint':native,
               'source_hashes':sources,'preserved_before':before,'preserved_after':after,'preserved_run_files':runs,
               'old_artifacts_preserved_in_place':True,'old_stop_token':stop_token,
               'original_absolute_wall_cap_retained':True,'test_scored':False}
 atomic(new/'supersession.json',supersession);atomic(root/'reports/VG_V2_SUPERSESSION.json',supersession)
 env=dict(os.environ);env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='4')
 log=(new/'pipeline.log').open('w');err=(new/'pipeline.stderr.log').open('w')
 job=subprocess.Popen([sys.executable,'-B',str(root/'scripts/remote_vg_v2_speed_pipeline.py')],cwd=root,stdout=log,stderr=err,env=env,start_new_session=True)
 log.close();err.close();(new/'pipeline.pid').write_text(str(job.pid))
 time.sleep(3);assert job.poll() is None,'new_pipeline_early_exit'
 pointer={'experiment_id':cfg['experiment_id'],'artifact_root':'artifacts/vg_method_v2_speed_m2_v1',
          'pipeline_token':'remote_vg_v2_speed_pipeline.py','progress_report':'VG_V2_SPEED_PROGRESS.json',
          'result_report':'VG_V2_SPEED_RESULT.json','config_path':'configs/vg_method_v2_speed_m2_v1.json',
          'original_experiment_preserved':True,'preregistration_sha256':stop_token}
 atomic(root/'reports/VG_ACTIVE_EXPERIMENT.json',pointer)
 out.update(status='NEW_MATCHED_SPEED_EXPERIMENT_LAUNCHED',new_pid=job.pid,
            adopted_case='packed_fp32_m2',scientific_change_applied=True,
            all_four_arms_original_initialization=True,whole_epoch_speed_unmeasured=True)
except Exception as e:
 out.update(status='ADOPTION_NEEDS_REVIEW',reason=str(e),error_type=type(e).__name__,owned_stop_left_in_place=created_stop,traceback=traceback.format_exc()[-3000:])
finally:
 out['utc']=datetime.now(timezone.utc).isoformat();atomic(root/'reports/VG_SPEEDUP_ADOPTION.json',out)
 print(json.dumps(out))
