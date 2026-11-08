"""Once-only checkpoint-boundary benchmark worker with audited original resume.

Waits without an LLM. Stops only the pinned VG v2 via its cooperative STOP flag,
preserves all evidence, runs one bounded benchmark and restores original v2.
It never applies a scientific batching change without a new matched protocol.
"""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,shutil,signal,subprocess,sys,time,traceback
root=Path.home()/'rsg-former';art=root/'artifacts/vg_method_v2'
folder=root/'artifacts/caption_readiness/vg_speedup_controller_v1'
protocol=json.loads((folder/'preregistration.json').read_text())
started=time.monotonic();pid=protocol['v2_pid'];stop=art/'STOP';created_stop=False
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verify():
    assert all(sha(root/p)==h for p,h in protocol['source_hashes'].items()),'immutable_source_lock_changed'
def alive():
    p=Path('/proc')/str(pid)/'cmdline'
    return p.exists() and b'remote_vg_v2_pipeline.py' in p.read_bytes()
def atomic(path,obj):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(obj,indent=2));temp.replace(path)
def event(stage,**kw):
    atomic(folder/'status.json',{'utc':datetime.now(timezone.utc).isoformat(),'stage':stage,**kw})
def preserve(name):
    p=root/name
    if p.exists():
        destination=folder/'preserved'/name;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(p,destination)
        return {'path':name,'sha256':sha(destination),'bytes':destination.stat().st_size}
    return None
report={'status':'WAITING_NATIVE_CHECKPOINT','source_hashes':protocol['source_hashes'],
        'test_scored':False,'scientific_change_applied':False,'old_v2_sources_modified':False}
try:
    verify();event('waiting_native_checkpoint',pid=pid,poll_seconds=2)
    native=art/'screening_native_1729/report.json'
    while True:
        assert time.monotonic()-started<4*3600,'checkpoint_wait_timeout_no_stop'
        assert alive(),'v2_exited_before_checkpoint_no_restart'
        assert not stop.exists(),'preexisting_stop_no_override'
        if native.exists() and json.loads(native.read_text()).get('status')=='complete':break
        time.sleep(2)
    verify()
    done=json.loads(native.read_text())
    assert done['status']=='complete' and not done.get('test_scored',False)
    ckpt=root/done['checkpoint'];assert sha(ckpt)==done['checkpoint_sha256']
    before=[preserve(p) for p in ['reports/VG_V2_PREREGISTRATION.json','reports/VG_V2_PROGRESS.json','artifacts/vg_method_v2/screening_native_1729/report.json']]
    # The existing job checks this flag every image; no other process is signalled.
    with stop.open('x') as f:f.write(protocol['controller_lock_sha256'])
    created_stop=True;event('cooperative_stop_requested',pid=pid,selected_native_epoch=done['selected_epoch'])
    deadline=time.monotonic()+90
    while alive():
        assert time.monotonic()<deadline,'cooperative_stop_timeout_no_force_kill'
        time.sleep(.5)
    verify()
    outcome=json.loads((root/'reports/VG_V2_RESULT.json').read_text())
    assert outcome.get('reason')=='user_stop_flag' and not outcome.get('test_scored'), 'stop_outcome_requires_review'
    after=[preserve(p) for p in ['reports/VG_V2_RESULT.json','reports/VG_V2_PROGRESS.json','artifacts/vg_method_v2/pipeline.log']]
    report.update(status='BENCHMARK_RUNNING',native_completed=done,preserved_before=before,preserved_after=after)
    atomic(folder/'outcome.json',report);event('benchmark_running')
    env=dict(os.environ);env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1')
    log=(folder/'benchmark.stdout.log').open('w');err=(folder/'benchmark.stderr.log').open('w')
    proc=subprocess.Popen([sys.executable,'-B',str(root/'scripts/remote_vg_speedup_benchmark.py')],cwd=root,env=env,stdout=log,stderr=err,start_new_session=True)
    failure=None;deadline=time.monotonic()+250
    while proc.poll() is None:
        if time.monotonic()>deadline:failure='benchmark_wall_cap'
        try:
            rss=[l for l in (Path('/proc')/str(proc.pid)/'status').read_text().splitlines() if l.startswith('VmRSS:')]
            if rss and int(rss[0].split()[1])*1024>4*1024**3:failure='benchmark_cpu_rss_cap'
        except FileNotFoundError:pass
        if failure:
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=5)
            break
        time.sleep(.2)
    log.close();err.close()
    benchmark=root/'reports/VG_SPEEDUP_BENCHMARK.json'
    if benchmark.exists():report['benchmark']=json.loads(benchmark.read_text())
    else:
        report['benchmark']={'status':'BENCHMARK_FAILED','reason':failure or 'child_exit_without_outcome','exit_code':proc.returncode}
        atomic(benchmark,report['benchmark'])
    verify()
    assert not alive(),'old_process_unexpectedly_alive'
    assert stop.read_text()==protocol['controller_lock_sha256'],'stop_flag_ownership_changed'
    # A later arm might start in the two-second checkpoint polling interval.
    # Archive its partial run intact and restart that arm from original init.
    # Completed runs are never moved or rerun; no best checkpoint resume is assumed.
    archived=[]
    for name in ['screening_context_1729','screening_oneway_1729','screening_feedback_1729']:
        p=art/name
        if not p.exists():continue
        rp=p/'report.json'
        if rp.exists() and json.loads(rp.read_text()).get('status')=='complete':continue
        assert p.is_dir() and not p.is_symlink() and p.resolve().parent==art.resolve()
        destination=folder/'partial_runs'/name;destination.parent.mkdir(parents=True,exist_ok=True)
        assert not destination.exists();p.rename(destination);archived.append(name)
    resume={'utc':datetime.now(timezone.utc).isoformat(),'source_hashes':protocol['source_hashes'],
            'reason':'one authorized speed benchmark at completed native boundary',
            'native_completed_checkpoint_sha256':done['checkpoint_sha256'],
            'partial_runs_archived':archived,'resume_rule':'original locked pipeline skips completed native; any interrupted subsequent arm restarts from original initialization, never best/partial weights',
            'scientific_configuration_changed':False,'test_scored':False}
    atomic(folder/'audited_resume.json',resume)
    # Preserve STOP outcome before allowing original pipeline to generate its next result.
    (root/'reports/VG_V2_RESULT.json').rename(folder/'preserved_stop_result.json')
    stop.unlink();created_stop=False
    output=(art/'pipeline.log').open('a');diag=(folder/'resume.stderr.log').open('a')
    job=subprocess.Popen([sys.executable,'-B',str(root/'scripts/remote_vg_v2_pipeline.py')],cwd=root,stdout=output,stderr=diag,env=env,start_new_session=True)
    (art/'pipeline.pid').write_text(str(job.pid));output.close();diag.close()
    time.sleep(2);assert job.poll() is None,'original_resume_failed_to_start'
    budget_log=(folder/'original_budget_guard.log').open('a')
    budget=subprocess.Popen([sys.executable,'-B',str(root/'scripts/remote_vg_original_budget_guard.py'),str(job.pid)],cwd=root,stdout=budget_log,stderr=budget_log,env=env,start_new_session=True)
    budget_log.close();(folder/'budget_guard.pid').write_text(str(budget.pid))
    report.update(status='BENCHMARK_FINISHED_ORIGINAL_V2_RESUMED',resume=resume,resumed_pid=job.pid,
                  budget_guard_pid=budget.pid,original_wall_deadline_unix=protocol['original_wall_deadline_unix'],
                  next_action='If adoption_ready, preregister new matched four-arm experiment before changing microbatch; benchmark alone does not establish validation accuracy or superiority.')
    event('original_v2_resumed',pid=job.pid,benchmark_status=report['benchmark']['status'])
except Exception as e:
    report.update(status='CONTROLLER_NEEDS_REVIEW',error_type=type(e).__name__,reason=str(e),traceback=traceback.format_exc()[-3000:],owned_stop_left_in_place=created_stop)
    event('needs_review',reason=str(e))
finally:
    report['utc']=datetime.now(timezone.utc).isoformat()
    atomic(folder/'outcome.json',report);atomic(root/'reports/VG_SPEEDUP_CONTROLLER.json',report)
