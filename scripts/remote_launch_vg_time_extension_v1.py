"""Once-only time extension after completed screening; no old sources overwritten."""
from pathlib import Path
from datetime import datetime,timezone
import ast,hashlib,json,os,shutil,subprocess,sys,time

root=Path.home()/'rsg-former';art=root/'artifacts/vg_method_v2_speed_m2_v1'
folder=art/'time_extension_v1'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def atomic(p,obj):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(obj,indent=2));t.replace(p)
def alive(pid,token):
    p=Path('/proc')/str(pid)/'cmdline'
    return p.exists() and token.encode() in p.read_bytes()
assert not folder.exists(),'no_duplicate_time_extension'
old_lock=root/'reports/VG_V2_SPEED_PREREGISTRATION.json'
assert sha(old_lock)=='d5866565e34555143af0f2f42c240841eaa8e1f2b0152c05c4ef60121561377c'
old=json.loads(old_lock.read_text());cfg_path=root/'configs/vg_method_v2_time_extension_v1.json'
cfg=json.loads(cfg_path.read_text());before=old['config']
assert cfg['experiment_id']==before['experiment_id']=='vg_method_v2_speed_m2_v1'
assert cfg['execution_revision']=='vg_method_v2_time_extension_v1'
for key in before:
    if key!='resource':assert cfg[key]==before[key],key
assert {k:v for k,v in cfg['resource'].items() if k!='wall_hours'}=={k:v for k,v in before['resource'].items() if k!='wall_hours'}
assert cfg['resource']['wall_hours']==840
assert cfg['authorized_wall_deadline_unix']==cfg['original_wall_deadline_unix']+600*3600
pid=int((art/'pipeline.pid').read_text())
assert pid==2366909 and not alive(pid,'remote_vg_v2_speed_pipeline.py'),'old_pipeline_still_active'
assert not (art/'STOP').exists(),'existing_stop_no_override'
result=json.loads((root/'reports/VG_V2_SPEED_RESULT.json').read_text())
assert result['status']=='confirmation_resource_admission_blocked' and result['candidate']=='feedback' and not result['test_scored']
assert not (art/'test_lock.json').exists() and not (art/'features_test').exists(),'test_already_opened'
assert not any(p.is_dir() for p in art.glob('confirmation_*')),'confirmation_already_started'
inherited={}
for v in before['screening']['variants']:
    report_path=art/('screening_'+v+'_1729')/'report.json'
    row=json.loads(report_path.read_text())
    assert row==result['screening'][v] and row['status']=='complete' and len(row['history'])==6 and not row['test_scored']
    assert row['initial_head_sha256']=='bc2ceb642f57f26729b104da880f9e5057c3b92fac891e615515884d66efe09f'
    for p in [report_path,root/row['checkpoint'],report_path.parent/'best_validation_records.json']:
        inherited[p.relative_to(root).as_posix()]=sha(p)
    assert inherited[row['checkpoint']]==row['checkpoint_sha256']
sources=dict(old['source_hashes'])
for name in ['scripts/remote_vg_v2_time_extension_v1.py','scripts/remote_launch_vg_time_extension_v1.py','configs/vg_method_v2_time_extension_v1.json']:
    sources[name]=sha(root/name)
for name,h in sources.items():assert sha(root/name)==h,name
fixture=json.loads((root/'reports/VG_TIME_EXTENSION_V1_FIXTURE.json').read_text())
assert fixture['status']=='PASS' and fixture['new_pipeline_sha256']==sources['scripts/remote_vg_v2_time_extension_v1.py'] and fixture['new_config_sha256']==sources['configs/vg_method_v2_time_extension_v1.json']
ast.parse((root/'scripts/remote_vg_v2_time_extension_v1.py').read_text())
admission=json.loads((art/'confirmation_admission.json').read_text())
remaining=cfg['authorized_wall_deadline_unix']-time.time()
assert remaining>admission['required_seconds'],'extension_insufficient_resource_admission'
assert shutil.disk_usage(root).free>24*1024**3,'disk_reserve_admission'
q=subprocess.run(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
assert q.returncode==0 and int(q.stdout.strip().splitlines()[0])*1024**2>=9*1024**3,'shared_gpu_free_admission'
assert sha(root/'artifacts/vg_method_v2/features/manifest.json')==old['cache_manifest_sha256']
for name,h in old['frozen_baseline_source_hashes'].items():assert sha(root/name)==h
folder.mkdir()
preserved={}
for rel in ['reports/VG_ACTIVE_EXPERIMENT.json','reports/VG_V2_SPEED_RESULT.json','reports/VG_V2_SPEED_PROGRESS.json','reports/VG_V2_SPEED_PREREGISTRATION.json','artifacts/vg_method_v2_speed_m2_v1/confirmation_admission.json','artifacts/vg_method_v2_speed_m2_v1/pipeline.log','artifacts/vg_method_v2_speed_m2_v1/pipeline.stderr.log','artifacts/vg_method_v2_speed_m2_v1/pipeline.pid','artifacts/vg_method_v2_speed_m2_v1/initialization.json']:
    p=root/rel
    if p.exists():
        dest=folder/'preserved'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest);preserved[rel]=sha(dest)
lock=dict(old);lock.update(utc=datetime.now(timezone.utc).isoformat(),config=cfg,source_hashes=sources,inherited_screening_file_hashes=inherited,parent_preregistration_sha256=sha(old_lock),preserved_before=preserved,execution_revision=cfg['execution_revision'],authorization=cfg['time_extension_authorization'],scientific_protocol_changed=False,confirmation_starts_original_initialization=True,screening_not_retrained=True,test_scored=False)
atomic(folder/'preregistration.json',lock);atomic(root/'reports/VG_TIME_EXTENSION_V1_PREREGISTRATION.json',lock)
digest=sha(folder/'preregistration.json')
atomic(root/'reports/VG_V2_TIME_EXT_PROGRESS.json',{'utc':datetime.now(timezone.utc).isoformat(),'stage':'authorized_confirmation_starting','experiment_id':cfg['experiment_id'],'execution_revision':cfg['execution_revision'],'wall_seconds':time.time()-(before['original_wall_deadline_unix']-240*3600),'version_wall_seconds':0})
assert not (root/'reports/VG_V2_TIME_EXT_RESULT.json').exists(),'new_result_exists'
env=dict(os.environ);env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1',CUDA_CACHE_DISABLE='1')
log=(folder/'pipeline.log').open('x');err=(folder/'pipeline.stderr.log').open('x')
job=subprocess.Popen([sys.executable,'-B',str(root/'scripts/remote_vg_v2_time_extension_v1.py')],cwd=root,stdout=log,stderr=err,env=env,start_new_session=True)
log.close();err.close();(folder/'pipeline.pid').write_text(str(job.pid))
time.sleep(3)
out={'utc':datetime.now(timezone.utc).isoformat(),'status':'TIME_EXTENSION_PIPELINE_LAUNCHED' if job.poll() is None else 'TIME_EXTENSION_EARLY_EXIT','new_pid':job.pid,'original_pid':pid,'execution_revision':cfg['execution_revision'],'preregistration_sha256':digest,'original_preregistration_sha256':sha(old_lock),'original_wall_deadline_unix':before['original_wall_deadline_unix'],'authorized_wall_deadline_unix':cfg['authorized_wall_deadline_unix'],'total_accounted_wall_hours':840,'remaining_seconds_at_launch':remaining,'required_seconds_at_admission':admission['required_seconds'],'scientific_protocol_changed':False,'screening_not_retrained':True,'other_resource_caps_unchanged':True,'test_scored':False}
if job.poll() is None:
    pointer={'experiment_id':cfg['experiment_id'],'execution_revision':cfg['execution_revision'],'artifact_root':'artifacts/vg_method_v2_speed_m2_v1','pipeline_pid_path':'artifacts/vg_method_v2_speed_m2_v1/time_extension_v1/pipeline.pid','pipeline_token':'remote_vg_v2_time_extension_v1.py','progress_report':'VG_V2_TIME_EXT_PROGRESS.json','result_report':'VG_V2_TIME_EXT_RESULT.json','config_path':'configs/vg_method_v2_time_extension_v1.json','original_experiment_preserved':True,'preregistration_sha256':digest,'scientific_preregistration_sha256':sha(old_lock)}
    atomic(root/'reports/VG_ACTIVE_EXPERIMENT.json',pointer)
atomic(root/'reports/VG_TIME_EXTENSION_V1_LAUNCH.json',out)
print(json.dumps(out))
assert job.poll() is None,'pipeline_early_exit_preserved_logs_no_blind_retry'
