"""Audited cooperative stop and one user-authorized selected-control test phase."""
from pathlib import Path
from datetime import datetime,timezone
import ast,hashlib,json,os,shutil,subprocess,sys,time
ROOT=Path.home()/'rsg-former';OLD=ROOT/'artifacts/vg_method_v2_speed_m2_v1';ART=ROOT/'artifacts/vg_component_control_test_v1'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def atomic(p,x):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2));t.replace(p)
def alive(pid):
 p=Path('/proc')/str(pid)/'cmdline';return p.exists() and b'remote_vg_v2_time_extension_v1.py' in p.read_bytes()
assert not ART.exists(),'no_duplicate_component_phase'
assert not (ROOT/'reports/VG_COMPONENT_TEST_ACCESS_LEDGER.json').exists(),'test_previously_opened'
assert not (OLD/'test_lock.json').exists() and not (OLD/'features_test').exists(),'old_test_previously_opened'
lock_path=ROOT/'reports/VG_TIME_EXTENSION_V1_PREREGISTRATION.json';assert sha(lock_path)=='7e7a41ca5a0943e75534c80a18ada65138ad6657f6424f30ce605a7e71b3371e'
parent=json.loads(lock_path.read_text())
for p,h in parent['source_hashes'].items():assert sha(ROOT/p)==h,p
pid=int((OLD/'time_extension_v1/pipeline.pid').read_text());assert pid==2993084 and alive(pid)
assert not (OLD/'STOP').exists(),'existing_stop_no_override'
controls={}
for variant in ('native','context'):
 row=json.loads((OLD/('confirmation_'+variant+'_1729')/'report.json').read_text());assert row['status']=='complete' and len(row['history'])==13 and not row['test_scored'] and row['guardrail_met']
 assert sha(ROOT/row['checkpoint'])==row['checkpoint_sha256'];controls[variant]=row
chosen=max(controls,key=lambda v:controls[v]['selected_metrics']['mR@50']);assert chosen=='context'
selected=controls[chosen];assert selected['selected_epoch']==10
assert selected['checkpoint_sha256']=='29017a69d7cb6d76d68c834b5d540712775a9e0cf9cae13efd17d78a7b69292e','human_frozen_checkpoint_changed'
sources=dict(parent['source_hashes'])
for p in ['scripts/remote_vg_component_control_test_v1.py','scripts/remote_launch_vg_component_control_test_v1.py','scripts/remote_vg_component_control_status.py','src/rsgformer/vg_canonical_data.py','src/rsgformer/vg_canonical_evaluation.py','src/rsgformer/vg_native_postprocess.py','vendor/salience_sgg/models/salience_sggmodule.py','vendor/salience_sgg/evaluation/vg_sgg_evaluator.py','vendor/salience_sgg/lib/fpn/box_intersections_cpu/bbox.pyx','data/vg_canonical_audit/reltr_vg_annotations.zip']:
 sources[p]=sha(ROOT/p)
assert sources['data/vg_canonical_audit/reltr_vg_annotations.zip']=='56856b4386220f40d6b1b522a2ae199a8b4188765fd9ebb5321c1d0f4f2d7fbc'
assert sources['src/rsgformer/vg_native_postprocess.py']=='9c8e4ad679eabc528421edb972e328ddf0da5db49670291bbb2f2499b54cdb08'
for name in ['NATIVE_POSTPROCESS_CPU_PROBE.json','CANONICAL_EVALUATOR_CPU_PROBE.json']:assert json.loads((ROOT/'reports'/name).read_text())['status']=='PASS'
ast.parse((ROOT/'scripts/remote_vg_component_control_test_v1.py').read_text());assert shutil.disk_usage(ROOT).free>20*1024**3
ART.mkdir();audit=ART/'preserved_confirmation';audit.mkdir()
preserved={}
def preserve(rel,when='before'):
 p=ROOT/rel
 if p.exists():
  dest=audit/when/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest);preserved[when+':'+rel]=sha(dest)
for p in ['reports/VG_ACTIVE_EXPERIMENT.json','reports/VG_V2_TIME_EXT_PROGRESS.json','reports/VG_TIME_EXTENSION_V1_PREREGISTRATION.json','artifacts/vg_method_v2_speed_m2_v1/time_extension_v1/pipeline.log','artifacts/vg_method_v2_speed_m2_v1/time_extension_v1/pipeline.stderr.log']:
 preserve(p)
manifest={p.relative_to(ROOT).as_posix():sha(p) for folder in OLD.glob('confirmation_*') if folder.is_dir() for p in folder.iterdir() if p.is_file()}
now=time.time();lock={'utc':datetime.now(timezone.utc).isoformat(),'phase_id':'vg_component_control_test_v1','authorization':'Human19Oct2026 explicitly requests selecting strongest validation control, locking it now, testing current component and comparing SOTA before full3seed confirmation. Supersedes previous3seed/+1pp test gate for this single exploratory component assessment only. No regional-superiority claim.','parent_preregistration_sha256':sha(lock_path),'selection_pool':'completed native/context confirmation13epoch seed1729; maximize selected validation mR50 with existing R guard; no test-based selection','selection_pool_reports':controls,'checkpoint':selected['checkpoint'],'checkpoint_sha256':selected['checkpoint_sha256'],'selection_validation_metrics':selected['selected_metrics'],'selected_variant':'context','selected_seed':1729,'selected_epoch':10,'source_hashes':sources,'cache_manifest_sha256':parent['cache_manifest_sha256'],'deadline_unix':min(now+36*3600,parent['config']['authorized_wall_deadline_unix']),'resource':{'allocator_gib':6,'process_gpu_gib':8,'rss_gib':4,'cpu':4,'reserve_gib':16,'aggregate_artifacts_gib':40,'component_artifacts_gib':2},'validation_gate':'reproduce selection legacy scores on5000VAL <=1e-8;32real-output upstream postprocessor parity; full canonical evaluator vs independent float64 set-matching <=1e-12. No calibration/search; mismatch blocks test.','test_policy':'single checkpoint, single26446 canonical TEST inference pass after validation gates; no retry/resume after test opened; no downstream tuning or model selection from test','comparability_limitations':['single validation-selected seed; not3seed regional-method confirmation','pretrained SGG-head adaptation and fixed-transform training differs from native full recipe','current input resize uses component rounding convention','canonical evaluation targets/postprocessor/evaluator; not a native-training reproduction','benchmark images may overlap external caption tests; not caption-clean'],'old_confirmation_run_files':manifest,'preserved_before_stop':preserved,'test_scored':False}
atomic(ART/'preregistration.json',lock);atomic(ROOT/'reports/VG_COMPONENT_CONTROL_TEST_PREREGISTRATION.json',lock);token=sha(ART/'preregistration.json')
with (OLD/'STOP').open('x') as f:f.write('human19Oct_component_test_supersession:'+token)
limit=time.monotonic()+90
while alive(pid):
 assert time.monotonic()<limit,'cooperative_stop_timeout_no_kill'
 time.sleep(.5)
outcome=json.loads((ROOT/'reports/VG_V2_TIME_EXT_RESULT.json').read_text());assert outcome.get('reason')=='user_stop_flag' and not outcome.get('test_scored')
for p in ['reports/VG_V2_TIME_EXT_RESULT.json','reports/VG_V2_TIME_EXT_PROGRESS.json','artifacts/vg_method_v2_speed_m2_v1/time_extension_v1/pipeline.log','artifacts/vg_method_v2_speed_m2_v1/time_extension_v1/pipeline.stderr.log']:
 preserve(p,'after')
final_manifest={p.relative_to(ROOT).as_posix():sha(p) for folder in OLD.glob('confirmation_*') if folder.is_dir() for p in folder.iterdir() if p.is_file()}
atomic(ART/'supersession.json',{'utc':datetime.now(timezone.utc).isoformat(),'status':'HUMAN_REQUESTED_COMPONENT_TEST_BEFORE_REMAINING_CONFIRMATION','old_pid':pid,'old_outcome':outcome,'preserved':preserved,'final_run_file_hashes':final_manifest,'old_run_files_preserved_in_place':True,'old_owned_stop_kept':True,'no_other_job_stopped':True,'test_scored':False})
q=subprocess.run(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
assert q.returncode==0 and int(q.stdout.splitlines()[0])*1024**2>=9*1024**3,'gpu_admission_blocked_after_owned_stop'
env=dict(os.environ);env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1',CUDA_CACHE_DISABLE='1')
log=(ART/'pipeline.log').open('x');err=(ART/'pipeline.stderr.log').open('x');job=subprocess.Popen([sys.executable,'-B',str(ROOT/'scripts/remote_vg_component_control_test_v1.py')],cwd=ROOT,env=env,stdout=log,stderr=err,start_new_session=True);log.close();err.close();(ART/'pipeline.pid').write_text(str(job.pid));time.sleep(3)
launch={'utc':datetime.now(timezone.utc).isoformat(),'status':'COMPONENT_VALIDATION_THEN_TEST_LAUNCHED' if job.poll() is None else 'COMPONENT_EARLY_EXIT','pid':job.pid,'old_pid_stopped':pid,'selected_variant':'context','selected_epoch':10,'seed':1729,'checkpoint_sha256':selected['checkpoint_sha256'],'preregistration_sha256':token,'test_scored':False,'old_confirmation_superseded_by_human':True}
atomic(ROOT/'reports/VG_COMPONENT_CONTROL_TEST_LAUNCH.json',launch);print(json.dumps(launch));assert job.poll() is None,'component_early_exit_preserved_no_blind_retry'
