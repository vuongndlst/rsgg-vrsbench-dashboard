"""Serial registered TEST identity indexing; no model or caption scoring."""
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '1'
from pathlib import Path
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import resource
import shutil
import subprocess
import sys
import time

ROOT = Path.home() / 'rsg-former'
PHASE = ROOT / 'artifacts/caption_readiness/oi_stream_phase_v1'
PREREG = PHASE / 'preregistration.json'
PROGRESS = ROOT / 'reports/OI_STREAM_FINGERPRINT_PROGRESS.json'
RESULT = ROOT / 'reports/OI_STREAM_PHASE_RESULT.json'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)

def decision(progress, resource_caps):
    """Terminal rules preserve every target ID, including failed source IDs."""
    assert progress['target_images'] == 125436
    assert 0 <= progress['indexed_images'] <= 125436
    assert progress['persistent_network_charge_bytes'] <= resource_caps['cumulative_network_charge_bytes']
    assert progress['disk_pixel_bytes_written'] == 0
    for key in ['benchmark_test_scored', 'model_or_metric_run', 'reference_captions_read',
                'oi_train_or_val_accessed', 'live_v2_modified', 'clean_release_certified']:
        assert progress[key] is False
    if progress['indexed_images'] == 125436:
        assert progress['status'] == 'OI_TEST_STREAM_INDEX_COMPLETE'
        assert progress['retry_exhausted_images'] == 0
        return 'pixel_index_complete'
    if progress['stop_reason'] == 'persistent_network_charge_cap':
        return 'stopped_network_budget'
    if progress['batch_images_processed'] == 0 and not progress['errors']:
        if progress['indexed_images'] + progress['retry_exhausted_images'] == 125436:
            return 'incomplete_source_retry_exhausted'
        return 'stopped_no_admissible_progress'
    return None

def main():
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    lock = (PHASE / 'pipeline.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    raw = PREREG.read_bytes()
    prereg = json.loads(raw)
    prereg_sha = hashlib.sha256(raw).hexdigest()
    assert prereg['scope'] == 'full Open Images TEST pixel identity audit only'
    assert not RESULT.exists(), 'terminal_phase_preserved_no_automatic_restart'
    caps = prereg['resource']
    started = time.monotonic()
    batches = 0
    previous_coverage = None
    outcome, reason, latest = None, None, None
    try:
        while outcome is None:
            assert digest(PREREG) == prereg_sha, 'preregistration_changed'
            for relative, expected in prereg['source_hashes'].items():
                assert digest(ROOT / relative) == expected, 'source_drift:'+relative
            if time.monotonic() - started >= caps['phase_wall_seconds']:
                outcome = 'stopped_wall_budget'
                break
            assert shutil.disk_usage(ROOT).free > caps['minimum_free_disk_bytes'] + caps['maximum_journal_bytes']
            remaining = caps['phase_wall_seconds'] - (time.monotonic() - started)
            if remaining < caps['child_timeout_seconds']:
                outcome = 'stopped_wall_budget'
                break
            journal = ROOT / 'artifacts/caption_readiness/oi_stream_fingerprints_full_v1.sqlite'
            if journal.exists():
                assert journal.stat().st_size < caps['maximum_journal_bytes']
            process = subprocess.run([sys.executable, '-B', str(ROOT / 'scripts/remote_oi_stream_fingerprint_batch.py')],
                                     cwd=ROOT, stdin=subprocess.DEVNULL, capture_output=True, text=True,
                                     timeout=caps['child_timeout_seconds'])
            batches += 1
            if process.returncode:
                outcome = 'stopped_batch_error'
                reason = (process.stderr or process.stdout)[-5000:]
                break
            latest = json.loads(PROGRESS.read_text())
            assert latest['source']['preregistration_sha256'] == prereg_sha
            assert latest['source']['script_sha256'] == prereg['source_hashes']['scripts/remote_oi_stream_fingerprint_batch.py']
            assert latest['peak_cpu_rss_bytes'] < caps['child_peak_rss_bytes']
            assert latest['journal_bytes'] < caps['maximum_journal_bytes']
            assert latest['batch_network_charge_bytes'] <= caps['batch_network_charge_bytes']
            coverage = (latest['indexed_images'], latest['retry_exhausted_images'],
                        latest['persistent_network_charge_bytes'])
            outcome = decision(latest, caps)
            if outcome is None and coverage == previous_coverage:
                outcome = 'stopped_no_admissible_progress'
            previous_coverage = coverage
            save(PHASE / 'status.json', {
                'utc':datetime.now(timezone.utc).isoformat(), 'status':'active' if outcome is None else outcome,
                'pid':os.getpid(), 'preregistration_sha256':prereg_sha,
                'batches':batches, 'elapsed_seconds':time.monotonic()-started,
                'indexed_images':latest['indexed_images'], 'target_images':125436,
                'retry_exhausted_images':latest['retry_exhausted_images'],
                'persistent_network_charge_bytes':latest['persistent_network_charge_bytes']})
    except subprocess.TimeoutExpired:
        # subprocess.run kills only the exact audit child it created. Its SQLite
        # reservation remains charged; the phase stops rather than hiding a retry.
        outcome, reason = 'stopped_batch_timeout', 'owned batch exceeded registered child timeout; preserve inflight charge'
    except Exception as error:
        outcome, reason = 'stopped_admission_or_provenance_error', type(error).__name__+': '+str(error)
    result = {'utc':datetime.now(timezone.utc).isoformat(), 'status':outcome, 'reason':reason,
              'preregistration_sha256':prereg_sha, 'batches':batches,
              'elapsed_seconds':time.monotonic()-started, 'last_progress':latest,
              'peak_parent_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
              'benchmark_test_scored':False, 'model_or_metric_run':False,
              'reference_captions_read':False, 'oi_train_or_val_accessed':False,
              'live_v2_modified':False, 'clean_release_certified':False,
              'next':'Preserve evidence; join complete protected identity snapshots and audit perceptual candidates before fresh clean release'}
    save(PHASE / 'outcome.json', result)
    save(RESULT, result)
    print(json.dumps(result))
    lock.close()

if __name__ == '__main__':
    main()
