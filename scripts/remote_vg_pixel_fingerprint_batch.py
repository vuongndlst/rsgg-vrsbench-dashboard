"""Resumable bounded VG image fingerprint index, direct ZIP reads only."""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import hashlib
import io
import json
import os
import shutil
import signal
import sqlite3
import struct
import time
import zipfile
for variable in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[variable] = '1'
import numpy as np
from PIL import Image

signal.alarm(100)
os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
started = time.monotonic()
root, vgroot = Path.home() / 'rsg-former', Path.home() / 'sbir/hienlth/vg'
folder = root / 'artifacts/caption_readiness'
assert shutil.disk_usage(root).free > 16 * 1024 ** 3 + 256 * 1024 ** 2
script_sha = hashlib.sha256((root / 'scripts/remote_vg_pixel_fingerprint_batch.py').read_bytes()).hexdigest()
annotation = root / 'data/vg_canonical_audit/reltr_vg_annotations.zip'
assert hashlib.sha256(annotation.read_bytes()).hexdigest() == '56856b4386220f40d6b1b522a2ae199a8b4188765fd9ebb5321c1d0f4f2d7fbc'
with zipfile.ZipFile(annotation) as bundle:
    roles = {int(item['id']): split for split in ['train', 'val', 'test']
             for item in json.loads(bundle.read('vg/' + split + '.json'))['images']}
assert len(roles) == 89169
archives = [zipfile.ZipFile(vgroot / name) for name in ['images.zip', 'images2.zip']]
members = {}
source_index = []
for index, archive in enumerate(archives):
    stat = Path(archive.filename).stat()
    for info in archive.infolist():
        stem = Path(info.filename).stem
        if not info.filename.lower().endswith('.jpg') or not stem.isdigit():
            continue
        identity = int(stem)
        assert identity not in members
        members[identity] = (index, info)
        source_index.append([index, identity, info.filename, info.file_size, info.compress_size, info.CRC])
source = {'script_sha256': script_sha, 'canonical_annotation_sha256': hashlib.sha256(annotation.read_bytes()).hexdigest(),
          'zip_central_member_sha256': hashlib.sha256(json.dumps(source_index, separators=(',', ':')).encode()).hexdigest(),
          'archives': [{'filename': Path(archive.filename).name, 'bytes': Path(archive.filename).stat().st_size,
                        'mtime_ns': Path(archive.filename).stat().st_mtime_ns} for archive in archives],
          'method': 'raw Pillow RGB/no EXIF; SHA256(width,height,RGB); normalized DCT32 low8 median excluding DC phash63; 0/90/180/270 rotations',
          'canonical_split_counts': dict(Counter(roles.values()))}
db = sqlite3.connect(folder / 'vg_pixel_fingerprints_v1.sqlite', timeout=5)
db.execute('CREATE TABLE IF NOT EXISTS provenance (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
db.execute('CREATE TABLE IF NOT EXISTS pixels (vg_id INTEGER PRIMARY KEY, split TEXT NOT NULL, jpeg_sha TEXT NOT NULL, rgb_sha TEXT NOT NULL, width INTEGER NOT NULL, height INTEGER NOT NULL, phash0 TEXT NOT NULL, phash90 TEXT NOT NULL, phash180 TEXT NOT NULL, phash270 TEXT NOT NULL, bytes INTEGER NOT NULL)')
provenance_raw = json.dumps(source, sort_keys=True, separators=(',', ':'))
existing = db.execute('SELECT value FROM provenance WHERE key="source"').fetchone()
if existing:
    assert existing[0] == provenance_raw, 'fingerprint_source_changed_new_journal_required'
else:
    db.execute('INSERT INTO provenance VALUES (?,?)', ('source', provenance_raw))
    db.commit()
done = {row[0] for row in db.execute('SELECT vg_id FROM pixels')}
assert done <= set(roles)
pending = sorted(set(roles) - done)
basis = np.cos(np.pi / 32 * np.arange(8)[:, None] * (np.arange(32)[None, :] + .5))
basis *= np.sqrt(2 / 32)
basis[0] /= np.sqrt(2)
processed, read_bytes = 0, 0
try:
    for identity in pending[:3000]:
        if time.monotonic() - started > 75:
            break
        index, info = members[identity]
        assert 0 < info.file_size <= 8 * 1024 ** 2
        if read_bytes + info.file_size > 768 * 1024 ** 2:
            break
        raw = archives[index].read(info)
        assert len(raw) == info.file_size
        read_bytes += len(raw)
        with Image.open(io.BytesIO(raw)) as image:
            assert image.width * image.height < 16_000_000
            rgb = image.convert('RGB')
            shape = [rgb.width, rgb.height]
            rgb_sha = hashlib.sha256(struct.pack('>II', *shape) + rgb.tobytes()).hexdigest()
            gray = np.asarray(rgb.convert('L').resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float64)
            hashes = []
            for rotation in range(4):
                low = (basis @ np.rot90(gray, rotation) @ basis.T).reshape(-1)[1:]
                bits = low > np.median(low)
                hashes.append(f'{sum(int(value) << bit for bit, value in enumerate(bits)):016x}')
        db.execute('INSERT INTO pixels VALUES (?,?,?,?,?,?,?,?,?,?,?)',
                   (identity, roles[identity], hashlib.sha256(raw).hexdigest(), rgb_sha,
                    *shape, *hashes, len(raw)))
        processed += 1
        if processed % 100 == 0:
            db.commit()
    db.commit()
    counts = dict(db.execute('SELECT split,COUNT(*) FROM pixels GROUP BY split').fetchall())
    total = sum(counts.values())
    report = {'utc': datetime.now(timezone.utc).isoformat(),
              'status': 'VG_PIXEL_INDEX_COMPLETE' if total == len(roles) else 'VG_PIXEL_INDEX_IN_PROGRESS',
              'batch_images_processed': processed, 'batch_zip_member_bytes_read': read_bytes,
              'batch_wall_seconds': time.monotonic() - started,
              'indexed_images': total, 'target_images': len(roles), 'indexed_by_canonical_split': counts,
              'source': source, 'journal_alias': '${REMOTE_HOME}/rsg-former/artifacts/caption_readiness/vg_pixel_fingerprints_v1.sqlite',
              'journal_bytes': (folder / 'vg_pixel_fingerprints_v1.sqlite').stat().st_size,
              'cpu_affinity_count': len(os.sched_getaffinity(0)), 'batch_image_cap': 3000,
              'vg_images_extracted': False, 'caption_reference_text_read': False,
              'model_inference_or_benchmark_scoring': False, 'live_v2_modified': False,
              'training_manifest_modified': False, 'clean_release_certified': False,
              'remaining': ['Complete resumable fingerprint coverage',
                            'Compare hashes/candidate images against protected VG and external held-out pixels',
                            'Audit perceptual candidates with an independently locked exclusion policy before final fresh release']}
    (root / 'reports/VG_PIXEL_FINGERPRINT_PROGRESS.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report))
finally:
    db.close()
    for archive in archives:
        archive.close()
