"""Admit one detached speed benchmark worker, preserving the live experiment."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,shutil,subprocess,sys
root=Path.home()/'rsg-former';art=root/'artifacts/vg_method_v2'
folder=root/'artifacts/caption_readiness/vg_speedup_controller_v1'
assert not folder.exists(),'no_duplicate_controller'
assert not (art/'STOP').exists(),'existing_stop_requires_review'
assert shutil.disk_usage(root).free>16*1024**3+256*1024**2
lock=json.loads((root/'reports/VG_V2_PREREGISTRATION.json').read_text())
sources=dict(lock['source_hashes'])
sources.update({'vendor/salience_sgg/models/salience_sgg/deformable_detr.py':'b79111a2ce07827915b37b6991e2904e2445ff701db30f7cac4be01d4783d67a',
                'vendor/salience_sgg/models/salience_sgg/deformable_transformer.py':'16ab624681c3a43686346cb07be8b0d178c7535ac78ff182c1ff3abe9ef27558'})
for p in ['src/rsgformer/vg_speedup.py','scripts/remote_vg_speedup_benchmark.py','scripts/remote_vg_speedup_controller.py','scripts/remote_launch_vg_speedup_controller.py','scripts/remote_vg_original_budget_guard.py']:
    sources[p]=hashlib.sha256((root/p).read_bytes()).hexdigest()
for p,h in sources.items():assert hashlib.sha256((root/p).read_bytes()).hexdigest()==h,p
fixture=json.loads((root/'reports/VG_SPEEDUP_FIXTURE.json').read_text());assert fixture['status']=='PASS'
assert fixture['module_sha256']==sources['src/rsgformer/vg_speedup.py'],'stale_fixture'
pid=int((art/'pipeline.pid').read_text())
assert b'remote_vg_v2_pipeline.py' in (Path('/proc')/str(pid)/'cmdline').read_bytes()
progress=json.loads((root/'reports/VG_V2_PROGRESS.json').read_text())
assert progress['run']=='screening_native_1729' and progress.get('epoch')==6
assert progress['stage'] in ['training','validation','epoch_complete']
protocol={'utc':datetime.now(timezone.utc).isoformat(),'v2_pid':pid,'source_hashes':sources,
          'authorization':'human 11Oct2026: ok vậy tiến hành nhé, referring to separate speed benchmark and parity checks',
          'stop_condition':'complete native screening report and verified selected checkpoint; cooperative STOP of exactly original v2',
          'wait_hours_cap':4,'poll_seconds':2,'benchmark_timeout_seconds':250,
          'post_benchmark':'audited original-pipeline resume, keeping native completed; archive interrupted next-arm artifacts, restart only that arm from original init',
          'no_other_process_signals':True,'no_live_source_changes':True,'no_test_access':True,
          'microbatch_adoption':'separate newly registered matched four-arm experiment only; never reuse old native as optimized control',
          'original_wall_budget_hours':240}
protocol['original_wall_deadline_unix']=datetime.fromisoformat(progress['utc']).timestamp()-progress['wall_seconds']+240*3600
protocol['controller_lock_sha256']=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
folder.mkdir();(folder/'preregistration.json').write_text(json.dumps(protocol,indent=2))
env=dict(os.environ);env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
out=(folder/'controller.log').open('w');err=(folder/'controller.stderr.log').open('w')
proc=subprocess.Popen([sys.executable,'-B',str(root/'scripts/remote_vg_speedup_controller.py')],cwd=root,env=env,stdout=out,stderr=err,start_new_session=True)
(folder/'controller.pid').write_text(str(proc.pid));out.close();err.close()
report={'utc':datetime.now(timezone.utc).isoformat(),'status':'SPEEDUP_CONTROLLER_LAUNCHED',
        'controller_pid':proc.pid,'v2_pid':pid,'live_job_stopped_now':False,
        'protocol':protocol,'preregistration_sha256':hashlib.sha256((folder/'preregistration.json').read_bytes()).hexdigest(),
        'scientific_change_applied':False,'benchmark_test_accessed':False}
(root/'reports/VG_SPEEDUP_LAUNCH.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
