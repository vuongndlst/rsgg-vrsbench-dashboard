"""Admit and preregister one CPU-only full TEST pixel identity audit."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, os, shutil, subprocess, sys, time
root = Path.home() / 'rsg-former'
os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
phase = root / 'artifacts/caption_readiness/oi_stream_phase_v1'
assert not phase.exists(), 'preserve_existing_phase_no_automatic_relaunch'
hashes = {
    'scripts/remote_oi_stream_fingerprint_batch.py':'752b93fc302851e5e1922bf0a918842f09fca29eade1136cf2b65a4d69f0e1dd',
    'scripts/remote_oi_stream_pipeline.py':'ebc038883d3e01b8bf81a023ea84059d5499a034a232dc3ea35b8ac9f7c127a1',
    'src/rsgformer/openimages_test_reader.py':'0bf8d41c541109e11c89e3e7dbaf47b38234a87a995baf2e5f2f2e0b16d307f6',
    'scripts/remote_oi_full_stream_ledger_probe_v2.py':'4eb3692aabae14259108ba39fdeb3da6bb31ad154cbf50e9a49360a8ce3dff5b'}
for relative, expected in hashes.items():
    assert hashlib.sha256((root/relative).read_bytes()).hexdigest() == expected, relative
hashes['scripts/remote_oi_stream_launch.py'] = hashlib.sha256((root/'scripts/remote_oi_stream_launch.py').read_bytes()).hexdigest()
fixture_raw = (root/'reports/OI_FULL_STREAM_LEDGER_PROBE.json').read_bytes()
fixture = json.loads(fixture_raw)
assert fixture['status']=='PASS' and fixture['external_owned_worker_timeout']
assert fixture['elapsed_seconds'] < 120 and fixture['network_downloaded_bytes']==0
assert not fixture['real_pixel_journals_modified'] and not fixture['live_v2_modified']
assert fixture['production_source_sha256']==hashes['scripts/remote_oi_stream_fingerprint_batch.py']
assert fixture['fixture_source_sha256']==hashes['scripts/remote_oi_full_stream_ledger_probe_v2.py']
assert all(value=='PASS' for value in fixture['cases'].values())
pilot_raw = (root/'reports/OI_STREAM_FINGERPRINT_PILOT.json').read_bytes()
pilot = json.loads(pilot_raw)
assert pilot['status']=='FIXED_STREAM_PILOT_COMPLETE' and pilot['indexed_images']==20
assert pilot['disk_pixel_bytes_written']==0 and not pilot['benchmark_test_scored']
v2 = json.loads((root/'reports/VG_V2_PREREGISTRATION.json').read_text())
for relative, expected in v2['source_hashes'].items():
    assert hashlib.sha256((root/relative).read_bytes()).hexdigest() == expected, relative
caps = {'cumulative_network_charge_bytes':64*1024**3, 'batch_network_charge_bytes':32*1024**2,
        'single_request_bytes':8*1024**2, 'max_attempts_per_id':2,
        'cpu_threads':1, 'cuda_used':False, 'child_peak_rss_bytes':768*1024**2,
        'child_timeout_seconds':110, 'batch_soft_seconds':75, 'batch_alarm_seconds':100,
        'phase_wall_seconds':96*3600, 'minimum_free_disk_bytes':16*1024**3,
        'maximum_journal_bytes':256*1024**2, 'disk_pixel_cache_bytes':0}
assert shutil.disk_usage(root).free > caps['minimum_free_disk_bytes'] + caps['maximum_journal_bytes']
v2_deadline = datetime.fromisoformat(v2['utc']).timestamp() + v2['config']['resource']['wall_hours']*3600
assert time.time()+caps['phase_wall_seconds'] < v2_deadline, 'independent_preparation_would_exceed_existing_program_window'
protocol = {'utc':datetime.now(timezone.utc).isoformat(),
            'scope':'full Open Images TEST pixel identity audit only',
            'source_hashes':hashes, 'resource':caps, 'target_images':125436,
            'identity_manifest_sha256':'6eb586b40bd68245e20205d896a5ae49700b72d61783b0e5ddc80a50e2242c87',
            'membership':'all sorted locked TEST IDs; fixed denominator; no human reference/model/score selection',
            'reader':'raw CVDF JPEG matrix, no EXIF transpose, no rotation/resize/cache writes',
            'retry_charge':'reserve8MiB committed before each attempt; failures/interruption charged8MiB; refund only successful fingerprint transaction; max2attempts per ID',
            'continuation':'serial bounded audit subprocesses; no monitor loop, GPU, scientific tuning or original v2 source/config changes; stop on provenance/admission error, budget, timeout, or no admissible work',
            'terminal_source_failures':'retain IDs and failure evidence; continue remaining admissible IDs; no complete or clean claim if any target missing',
            'journal_version':'oi_stream_fingerprints_full_v1.sqlite, separate from fixed20ID pilot and local-protected journal',
            'pilot_sha256':hashlib.sha256(pilot_raw).hexdigest(),
            'full_ledger_fixture_sha256':hashlib.sha256(fixture_raw).hexdigest(),
            'point_estimate_from20IDpilot':{'hours':pilot['wall_seconds']/20*125436/3600,
                                           'network_bytes':pilot['persistent_network_charge_bytes']/20*125436,
                                           'not_guaranteed':True},
            'existing_v2_source_hashes_preserved':v2['source_hashes'],
            'benchmark_test_scored':False, 'clean_release_certified':False,
            'full_image_cache_storage_gate_remains_failed':True,
            'next':'Join snapshots with protected VG identities/pixels, independently verify perceptual candidates, then version final clean release before fresh training'}
phase.mkdir()
raw = json.dumps(protocol,indent=2).encode()
(phase/'preregistration.json').write_bytes(raw)
prereg_sha = hashlib.sha256(raw).hexdigest()
(root/'reports/OI_STREAM_PHASE_PREREGISTRATION.json').write_bytes(raw)
environment = os.environ.copy()
environment.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
                   CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',CUDA_CACHE_DISABLE='1')
log = (phase/'pipeline.log').open('ab')
process = subprocess.Popen([sys.executable,'-B',str(root/'scripts/remote_oi_stream_pipeline.py')],
                           cwd=root, env=environment, stdin=subprocess.DEVNULL,
                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
log.close()
(phase/'pipeline.pid').write_text(str(process.pid))
time.sleep(1)
report = {'utc':datetime.now(timezone.utc).isoformat(), 'status':'launched' if process.poll() is None else 'exited_after_launch',
          'pid':process.pid, 'preregistration_sha256':prereg_sha, 'resource':caps,
          'source_hashes':hashes, 'target_images':125436,
          'benchmark_test_scored':False, 'reference_captions_read':False,
          'model_or_metric_run':False, 'live_v2_modified':False,
          'full_protected_pixel_coverage':False, 'clean_release_certified':False}
(root/'reports/OI_STREAM_PHASE_LAUNCH.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report))
