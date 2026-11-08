"""Bounded local held-out COCO/Flickr/OI fingerprints; no download or inference."""
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
for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[name] = '1'
import numpy as np
from PIL import Image

signal.alarm(100)
os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
started = time.monotonic()
root = Path.home() / 'rsg-former'
shared = Path.home() / 'shared-assets/datasets'
folder = root / 'artifacts/caption_readiness'
assert shutil.disk_usage(root).free > 16 * 1024 ** 3 + 256 * 1024 ** 2
script_sha = hashlib.sha256((root / 'scripts/remote_external_pixel_fingerprint_batch.py').read_bytes()).hexdigest()
sources, targets, present = {}, {}, {}
for dataset, filename, expected in [
        ('coco_karpathy', 'dataset_coco.json', '2fd999220673258012acfb411a4e7e66af7d488050b2519b0badcc49b7600b8d'),
        ('flickr30k_karpathy', 'dataset_flickr30k.json', 'db779ee2c5df40c25ff0eebeca464f6124b620cc4a9398650aea71fd1b53d4af')]:
    raw = (shared / dataset / 'annotations' / filename).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == expected
    sources[dataset] = expected
    directory = shared / dataset / 'images'
    existing = {path.name: path for path in directory.iterdir() if path.is_file()}
    # Only image split/filename/IDs are used; no sentences/reference text is retained.
    for image in json.loads(raw)['images']:
        if image['split'] not in ['val', 'test']:
            continue
        name = image['filename']
        assert Path(name).name == name
        identity = str(image['cocoid'] if dataset == 'coco_karpathy' else image['imgid'])
        key = (dataset, image['split'], identity)
        assert key not in targets
        targets[key] = name
        if name in existing:
            present[key] = existing[name]
oi_raw = (shared / 'openimages_test/test_identity_manifest.json').read_bytes()
sources['openimages_test'] = hashlib.sha256(oi_raw).hexdigest()
assert sources['openimages_test'] == '6eb586b40bd68245e20205d896a5ae49700b72d61783b0e5ddc80a50e2242c87'
oi = json.loads(oi_raw)
assert len(set(oi['caption_test_ids'])) == 125436
oi_dir = shared / 'openimages_test/images'
existing = {path.name: path for path in oi_dir.iterdir() if path.is_file()}
for identity in oi['caption_test_ids']:
    name = identity + '.jpg'
    key = ('openimages_test', 'test', identity)
    targets[key] = name
    if name in existing:
        present[key] = existing[name]
source = {'script_sha256': script_sha, 'metadata_sha256': sources,
          'method': 'raw Pillow RGB/no EXIF; SHA256(width,height,RGB); normalized DCT32 low8 median excluding DC phash63; 0/90/180/270 rotations',
          'target_membership_sha256': hashlib.sha256(json.dumps(sorted(targets), separators=(',', ':')).encode()).hexdigest()}
db = sqlite3.connect(folder / 'external_pixel_fingerprints_v1.sqlite', timeout=5)
db.execute('CREATE TABLE IF NOT EXISTS provenance (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
db.execute('CREATE TABLE IF NOT EXISTS pixels (dataset TEXT NOT NULL, split TEXT NOT NULL, image_id TEXT NOT NULL, jpeg_sha TEXT NOT NULL, rgb_sha TEXT NOT NULL, width INTEGER NOT NULL, height INTEGER NOT NULL, phash0 TEXT NOT NULL, phash90 TEXT NOT NULL, phash180 TEXT NOT NULL, phash270 TEXT NOT NULL, source_filename TEXT NOT NULL, bytes INTEGER NOT NULL, mtime_ns INTEGER NOT NULL, PRIMARY KEY(dataset,split,image_id))')
provenance_raw = json.dumps(source, sort_keys=True, separators=(',', ':'))
old = db.execute('SELECT value FROM provenance WHERE key="source"').fetchone()
if old:
    assert old[0] == provenance_raw, 'external_fingerprint_source_changed_new_journal_required'
else:
    db.execute('INSERT INTO provenance VALUES (?,?)', ('source', provenance_raw))
    db.commit()
done = set()
for dataset, split, identity, size, mtime in db.execute('SELECT dataset,split,image_id,bytes,mtime_ns FROM pixels'):
    key = (dataset, split, identity)
    assert key in targets and key in present, 'indexed_protected_pixel_disappeared'
    stat = present[key].stat()
    assert stat.st_size == size and stat.st_mtime_ns == mtime, 'indexed_protected_pixel_source_changed'
    done.add(key)
pending = sorted(set(present) - done)
basis = np.cos(np.pi / 32 * np.arange(8)[:, None] * (np.arange(32)[None, :] + .5))
basis *= np.sqrt(2 / 32)
basis[0] /= np.sqrt(2)
processed, read_bytes = 0, 0
try:
    for key in pending[:3000]:
        if time.monotonic() - started > 75:
            break
        path = present[key]
        stat = path.stat()
        assert 0 < stat.st_size <= 8 * 1024 ** 2
        if read_bytes + stat.st_size > 768 * 1024 ** 2:
            break
        raw = path.read_bytes()
        assert len(raw) == stat.st_size
        read_bytes += len(raw)
        with Image.open(io.BytesIO(raw)) as image:
            assert image.format == 'JPEG' and image.width * image.height < 16_000_000
            rgb = image.convert('RGB')
            shape = [rgb.width, rgb.height]
            rgb_sha = hashlib.sha256(struct.pack('>II', *shape) + rgb.tobytes()).hexdigest()
            gray = np.asarray(rgb.convert('L').resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float64)
            hashes = []
            for rotation in range(4):
                low = (basis @ np.rot90(gray, rotation) @ basis.T).reshape(-1)[1:]
                hashes.append(f'{sum(int(value) << bit for bit, value in enumerate(low > np.median(low))):016x}')
        assert path.stat().st_mtime_ns == stat.st_mtime_ns
        db.execute('INSERT INTO pixels VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                   (*key, hashlib.sha256(raw).hexdigest(), rgb_sha, *shape, *hashes,
                    path.name, len(raw), stat.st_mtime_ns))
        processed += 1
        if processed % 100 == 0:
            db.commit()
    db.commit()
    counts = {dataset + '/' + split: count for dataset, split, count in
              db.execute('SELECT dataset,split,COUNT(*) FROM pixels GROUP BY dataset,split')}
    total = sum(counts.values())
    report = {'utc': datetime.now(timezone.utc).isoformat(),
              'status': 'LOCAL_PROTECTED_PIXELS_INDEX_IN_PROGRESS' if total < len(present) else 'LOCAL_PROTECTED_PIXELS_INDEXED_PARTIAL_PROTECTED_COVERAGE',
              'batch_images_processed': processed, 'batch_pixel_bytes_read': read_bytes,
              'batch_wall_seconds': time.monotonic() - started, 'indexed_images': total,
              'target_protected_images': len(targets), 'existing_local_protected_pixels': len(present),
              'missing_local_protected_pixels': len(targets) - len(present),
              'indexed_by_dataset_split': counts,
              'target_by_dataset_split': dict(Counter(dataset + '/' + split for dataset, split, _ in targets)),
              'existing_by_dataset_split': dict(Counter(dataset + '/' + split for dataset, split, _ in present)),
              'source': source, 'journal_alias': '${REMOTE_HOME}/rsg-former/artifacts/caption_readiness/external_pixel_fingerprints_v1.sqlite',
              'journal_bytes': (folder / 'external_pixel_fingerprints_v1.sqlite').stat().st_size,
              'network_downloaded_bytes': 0, 'image_pixels_written': False,
              'caption_reference_text_used': False, 'model_or_metric_run': False,
              'benchmark_test_scored': False, 'live_v2_modified': False,
              'full_protected_pixel_coverage_audited': False, 'clean_release_certified': False,
              'remaining': ['Finish existing local pixel index and compare with VG exact/perceptual candidates',
                            'Obtain remaining authorized protected pixels or bounded streaming fingerprint coverage',
                            'No claim of a fully clean training release from partial coverage']}
    (root / 'reports/EXTERNAL_PIXEL_FINGERPRINT_PROGRESS.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report))
finally:
    db.close()
