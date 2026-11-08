"""Registered full TEST pixel audit, bounded resumable serial batches.

Separate journal/version from the fixed pilot. No model, references or cache.
"""
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '1'
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import fcntl
import json
import resource
import shutil
import signal
import sqlite3
import struct
import sys
import time
import numpy as np
from PIL import Image

root = Path.home() / 'rsg-former'
sys.path.insert(0, str(root / 'src'))
os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
started = time.monotonic()

def timeout_handler(signum, frame):
    raise TimeoutError('pilot_wall_alarm100s')

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(100)
assert shutil.disk_usage(root).free > 16 * 1024 ** 3 + 64 * 1024 ** 2
reader_path = root / 'src/rsgformer/openimages_test_reader.py'
reader_sha = hashlib.sha256(reader_path.read_bytes()).hexdigest()
assert reader_sha == '0bf8d41c541109e11c89e3e7dbaf47b38234a87a995baf2e5f2f2e0b16d307f6'
from rsgformer.openimages_test_reader import OpenImagesTestReader
shared = Path.home() / 'shared-assets/datasets/openimages_test'
manifest_sha = '6eb586b40bd68245e20205d896a5ae49700b72d61783b0e5ddc80a50e2242c87'
phase = root / 'artifacts/caption_readiness/oi_stream_phase_v1'
prereg_raw = (phase / 'preregistration.json').read_bytes()
prereg = json.loads(prereg_raw)
self_sha = hashlib.sha256((root / 'scripts/remote_oi_stream_fingerprint_batch.py').read_bytes()).hexdigest()
assert prereg['source_hashes']['scripts/remote_oi_stream_fingerprint_batch.py'] == self_sha
assert prereg['scope'] == 'full Open Images TEST pixel identity audit only'
assert prereg['resource']['cumulative_network_charge_bytes'] == 64 * 1024 ** 3
single_cap, cumulative_cap = 8 * 1024 ** 2, 64 * 1024 ** 3
batch_charge_cap = 32 * 1024 ** 2
reader = OpenImagesTestReader(shared, manifest_sha256=manifest_sha, allow_network=True,
                             single_image_bytes_cap=single_cap, network_bytes_cap=batch_charge_cap)
ids = reader.ids
assert len(ids) == 125436
source = {'script_sha256': self_sha,
          'preregistration_sha256': hashlib.sha256(prereg_raw).hexdigest(),
          'reader_sha256': reader_sha, 'identity_manifest_sha256': manifest_sha,
          'membership_sha256': hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest(),
          'selection': 'all sorted125436 locked TEST IDs, no reference/score selection',
          'method': 'raw Pillow RGB/no EXIF; SHA256(width,height,RGB); normalized DCT32 low8 median excluding DC phash63; 0/90/180/270 rotations',
          'caps': {'target_images': 125436, 'cumulative_network_charge_bytes': cumulative_cap,
                   'batch_network_charge_bytes': batch_charge_cap, 'batch_image_cap': 3000,
                   'single_request_bytes': single_cap, 'max_attempts_per_id': 2,
                   'batch_soft_seconds': 75, 'alarm_seconds': 100, 'cpu_threads': 1,
                   'peak_rss_gate_bytes': 768 * 1024 ** 2},
          'failed_or_interrupted_request_charge': 'full single-image cap; reservation committed before network; refund only after successful decode/fingerprint'}
folder = root / 'artifacts/caption_readiness'
folder.mkdir(exist_ok=True)
db_path = folder / 'oi_stream_fingerprints_full_v1.sqlite'
lock = (folder / 'oi_stream_fingerprints_full_v1.lock').open('a')
fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
db = sqlite3.connect(db_path, timeout=5)
db.execute('CREATE TABLE IF NOT EXISTS provenance (key TEXT PRIMARY KEY,value TEXT NOT NULL)')
db.execute('CREATE TABLE IF NOT EXISTS attempts (seq INTEGER PRIMARY KEY,image_id TEXT NOT NULL,state TEXT NOT NULL,charged_bytes INTEGER NOT NULL,actual_network_bytes INTEGER,reason TEXT,utc TEXT NOT NULL)')
db.execute('CREATE TABLE IF NOT EXISTS pixels (image_id TEXT PRIMARY KEY,jpeg_sha TEXT NOT NULL,rgb_sha TEXT NOT NULL,width INTEGER NOT NULL,height INTEGER NOT NULL,phash0 TEXT NOT NULL,phash90 TEXT NOT NULL,phash180 TEXT NOT NULL,phash270 TEXT NOT NULL,provenance_json TEXT NOT NULL)')
db.execute('CREATE INDEX IF NOT EXISTS attempt_image_idx ON attempts(image_id)')
raw_source = json.dumps(source, sort_keys=True, separators=(',', ':'))
previous = db.execute('SELECT value FROM provenance WHERE key=?', ('source',)).fetchone()
if previous:
    assert previous[0] == raw_source, 'pilot_source_changed_new_version_required'
else:
    db.execute('INSERT INTO provenance VALUES (?,?)', ('source', raw_source))
    db.commit()
assert set(row[0] for row in db.execute('SELECT image_id FROM pixels')).issubset(ids)
# A crash never refunds unknown network traffic. It consumes one attempt.
db.execute("UPDATE attempts SET state='interrupted',reason='previous process exited with request in flight' WHERE state='inflight'")
db.commit()
batch_start_charge = db.execute('SELECT COALESCE(SUM(charged_bytes),0) FROM attempts').fetchone()[0]
basis = np.cos(np.pi / 32 * np.arange(8)[:, None] * (np.arange(32)[None, :] + .5))
basis *= np.sqrt(2 / 32)
basis[0] /= np.sqrt(2)
processed, batch_network = 0, 0
errors, stop = [], None
try:
    done = set(row[0] for row in db.execute('SELECT image_id FROM pixels'))
    for identity in ids:
        if identity in done:
            continue
        if time.monotonic() - started > 75 or processed >= 3000:
            stop = 'batch_soft_time_cap'
            break
        attempts = db.execute('SELECT COUNT(*) FROM attempts WHERE image_id=?', (identity,)).fetchone()[0]
        if attempts >= 2:
            continue
        charged = db.execute('SELECT COALESCE(SUM(charged_bytes),0) FROM attempts').fetchone()[0]
        # Reserve even for cached reads: source disappearance cannot evade admission.
        if charged + single_cap > cumulative_cap:
            stop = 'persistent_network_charge_cap'
            break
        if charged - batch_start_charge + single_cap > batch_charge_cap:
            stop = 'batch_network_charge_cap'
            break
        cursor = db.execute('INSERT INTO attempts(image_id,state,charged_bytes,utc) VALUES (?,?,?,?)',
                            (identity, 'inflight', single_cap, datetime.now(timezone.utc).isoformat()))
        seq = cursor.lastrowid
        db.commit()
        before = reader.downloaded_bytes
        item = None
        try:
            item = reader.read(identity)
            rgb = item.image
            assert rgb.mode == 'RGB'
            rgb_sha = hashlib.sha256(struct.pack('>II', rgb.width, rgb.height) + rgb.tobytes()).hexdigest()
            gray = np.asarray(rgb.convert('L').resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float64)
            hashes = []
            for rotation in range(4):
                low = (basis @ np.rot90(gray, rotation) @ basis.T).reshape(-1)[1:]
                hashes.append(f'{sum(int(value) << bit for bit, value in enumerate(low > np.median(low))):016x}')
            actual = reader.downloaded_bytes - before
            assert 0 <= actual <= single_cap
            db.execute('INSERT INTO pixels VALUES (?,?,?,?,?,?,?,?,?,?)',
                       (identity, item.provenance['sha256'], rgb_sha, rgb.width, rgb.height,
                        *hashes, json.dumps(item.provenance, sort_keys=True)))
            db.execute('UPDATE attempts SET state=?,charged_bytes=?,actual_network_bytes=? WHERE seq=?',
                       ('complete', actual, actual, seq))
            db.commit()
            batch_network += actual
            processed += 1
        except Exception as error:
            db.rollback()
            db.execute('UPDATE attempts SET state=?,reason=? WHERE seq=?', ('failed', type(error).__name__+': '+str(error), seq))
            db.commit()
            errors.append({'image_id': identity, 'error': type(error).__name__+': '+str(error)})
            if isinstance(error, TimeoutError):
                stop = 'wall_alarm'
                break
        finally:
            if item is not None:
                item.image.close()
    rows = list(db.execute('SELECT image_id,jpeg_sha,rgb_sha,width,height,phash0,phash90,phash180,phash270 FROM pixels ORDER BY image_id'))
    total_charge = db.execute('SELECT COALESCE(SUM(charged_bytes),0) FROM attempts').fetchone()[0]
    assert total_charge <= cumulative_cap
    assert total_charge - batch_start_charge <= batch_charge_cap
    exhausted = db.execute('SELECT COUNT(*) FROM (SELECT image_id FROM attempts GROUP BY image_id HAVING COUNT(*)>=2) a WHERE NOT EXISTS(SELECT 1 FROM pixels p WHERE p.image_id=a.image_id)').fetchone()[0]
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    assert peak < source['caps']['peak_rss_gate_bytes'], 'pilot_cpu_rss_gate'
    report = {'utc': datetime.now(timezone.utc).isoformat(),
              'status': 'OI_TEST_STREAM_INDEX_COMPLETE' if len(rows)==len(ids) else 'OI_TEST_STREAM_INDEX_IN_PROGRESS',
              'source': source, 'indexed_images': len(rows), 'target_images': len(ids),
              'full_test_target_images': len(reader.ids), 'batch_images_processed': processed,
              'batch_successful_network_bytes': batch_network,
              'batch_reader_counted_network_bytes': reader.downloaded_bytes,
              'persistent_network_charge_bytes': total_charge,
              'batch_network_charge_bytes': total_charge - batch_start_charge,
              'retry_exhausted_images': exhausted,
              'fingerprint_rows_sha256': hashlib.sha256(json.dumps(rows,separators=(',', ':')).encode()).hexdigest(),
              'journal_alias': '${REMOTE_HOME}/rsg-former/artifacts/caption_readiness/oi_stream_fingerprints_full_v1.sqlite',
              'journal_bytes': db_path.stat().st_size, 'peak_cpu_rss_bytes': peak,
              'wall_seconds': time.monotonic()-started, 'errors': errors, 'stop_reason': stop,
              'disk_pixel_bytes_written': 0, 'model_or_metric_run': False,
              'reference_captions_read': False, 'benchmark_test_scored': False,
              'oi_train_or_val_accessed': False, 'live_v2_modified': False,
              'full_protected_pixel_coverage': False, 'clean_release_certified': False,
              'remaining': ['Finish full TEST pixel identity coverage within registered phase budget',
                            'Handle terminal retry-exhausted source IDs explicitly without reducing denominator',
                            'Join audited streaming snapshots with VG pixel comparisons before final clean release']}
    assert db_path.stat().st_size < 256 * 1024 ** 2
    (root / 'reports/OI_STREAM_FINGERPRINT_PROGRESS.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report))
finally:
    db.close()
    lock.close()
