"""Deterministic TEST-only pixel download pilot for resource/decoder contracts.

Never runs a model or scores/tunes captions. Sample selection uses image IDs only.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
import random
import shutil
import threading
import time
import urllib.request
from PIL import Image

root = Path.home() / 'rsg-former'
shared = Path.home() / 'shared-assets/datasets/openimages_test'
images = shared / 'images'
images.mkdir(parents=True, exist_ok=True)
report = root / 'reports/OPENIMAGES_TEST_PIXEL_PILOT.json'
meta = json.loads((root / 'reports/OPENIMAGES_TEST_METADATA.json').read_text())
manifest = json.loads((shared / 'test_identity_manifest.json').read_text())
assert meta['status'] == 'complete'
assert hashlib.sha256((shared / 'test_identity_manifest.json').read_bytes()).hexdigest() == meta['identity_manifest_sha256']
ids = manifest['caption_test_ids']
selection = sorted(ids, key=lambda value: hashlib.sha256(('OI-pixel-resource-pilot-12773:' + value).encode()).digest())[:128]
records_by_id = {row['ImageID']: row for row in manifest['images']}
started = time.monotonic()
download_bytes = 0
lock = threading.Lock()
state = {'status': 'running', 'scope': 'Open Images TEST only, resource/header pilot',
         'sample_images': len(selection), 'selection': '128 lowest SHA256(seed12773 + image ID), no visual/caption criteria',
         'wall_seconds_cap': 360, 'download_bytes_cap': 128 * 2**20,
         'single_image_bytes_cap': 8 * 2**20, 'network_workers': 2,
         'model_inference': False, 'test_scored': False, 'prompt_tuned': False}

def save(status, **values):
    state.update(status=status, updated_utc=datetime.now(timezone.utc).isoformat(), **values)
    report.write_text(json.dumps(state, indent=2))

def guard():
    assert time.monotonic() - started <= 360, 'pixel pilot wall cap'
    assert shutil.disk_usage(root).free >= 16 * 2**30, 'disk reserve'

def one(identity):
    global download_bytes
    assert len(identity) == 16 and all(c in '0123456789abcdef' for c in identity)
    guard()
    path = images / (identity + '.jpg')
    url = 'https://open-images-dataset.s3.amazonaws.com/test/' + identity + '.jpg'
    if path.exists():
        raw = path.read_bytes()
        assert 0 < len(raw) <= 8 * 2**20
    else:
        raw = None
        for attempt in range(2):
            try:
                with urllib.request.urlopen(url, timeout=20) as response:
                    assert 'image' in response.headers.get('Content-Type', '')
                    declared = int(response.headers.get('Content-Length', '0'))
                    assert 0 < declared <= 8 * 2**20
                    raw = response.read(8 * 2**20 + 1)
                    assert len(raw) == declared
                break
            except Exception:
                if attempt == 1: raise
                guard()
        with lock:
            assert download_bytes + len(raw) <= 128 * 2**20, 'pixel pilot download cap'
            download_bytes += len(raw)
        with Image.open(io.BytesIO(raw)) as image:
            image.verify()
        partial = path.with_suffix('.jpg.part')
        partial.write_bytes(raw)
        partial.replace(path)
    with Image.open(io.BytesIO(raw)) as image:
        image.load()
        dimensions = list(image.size)
        mode = image.mode
    return {'image_id': identity, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(),
            'md5': hashlib.md5(raw).hexdigest(), 'dimensions': dimensions, 'mode': mode,
            'source': url, 'metadata_rotation': records_by_id[identity]['Rotation']}

try:
    save('downloading')
    # Save the exact official source snapshot used to derive the public bucket/key contract.
    source_url = 'https://raw.githubusercontent.com/openimages/dataset/master/downloader.py'
    with urllib.request.urlopen(source_url, timeout=20) as response: source = response.read(65536)
    assert b"BUCKET_NAME = 'open-images-dataset'" in source
    pinned = root / 'research/references/openimages'
    pinned.mkdir(parents=True, exist_ok=True)
    (pinned / 'downloader.py').write_bytes(source)
    with ThreadPoolExecutor(max_workers=2) as pool:
        records = list(pool.map(one, selection))
    sizes = [row['bytes'] for row in records]
    mean = sum(sizes) / len(sizes)
    rng = random.Random(12773)
    means = sorted(sum(rng.choices(sizes, k=len(sizes))) / len(sizes) for _ in range(1000))
    high_mean = means[974]
    v2 = root / 'artifacts/vg_method_v2'
    existing = sum(path.stat().st_size for path in v2.rglob('*') if path.is_file())
    cfg = json.loads((root / 'configs/vg_method_v2.json').read_text())
    remaining_v2_reserved = max(0, cfg['resource']['artifact_limit_gib'] * 2**30 - existing)
    free = shutil.disk_usage(root).free
    allowed = max(0, free - 16 * 2**30 - remaining_v2_reserved)
    upper_projection = math.ceil(high_mean * len(ids))
    total_projection = math.ceil(mean * len(ids))
    safety_projection = math.ceil(upper_projection * 1.25)
    save('complete', valid_images=len(records), downloaded_bytes=download_bytes,
         sampled_total_bytes=sum(sizes), mean_image_bytes=mean, max_sample_image_bytes=max(sizes),
         projected_full_test_bytes=total_projection, bootstrap_upper_mean_projection_bytes=upper_projection,
         safety_projection_bytes=safety_projection,
         projection_caveat='small ID-random resource sample, not a guaranteed bound; sequential download must enforce actual disk/byte caps',
         v2_existing_artifact_bytes=existing, v2_remaining_artifact_budget_reserved_bytes=remaining_v2_reserved,
         disk_free_bytes=free, additional_image_storage_with_v2_budget_and_reserve_bytes=allowed,
         projected_storage_gate_pass=safety_projection <= allowed,
         fresh_clean_program_storage_reserved=False,
         full_test_download_launched=False, full_test_pixels_complete=False,
         official_downloader_source_url=source_url,
         official_downloader_source_sha256=hashlib.sha256(source).hexdigest(),
         elapsed_seconds=time.monotonic()-started, records=records)
    print(json.dumps({key: value for key, value in state.items() if key != 'records'}, indent=2))
except Exception as error:
    save('failed', error_type=type(error).__name__, reason=str(error)[:240], downloaded_bytes=download_bytes)
    print(json.dumps({key: value for key, value in state.items() if key != 'records'}, indent=2))
    raise
