"""Bounded official Open Images TEST metadata and human reference preparation.

No train/validation assets, image pixels, model inference, prompt tuning, or scores.
Caption text stays in evaluation-only artifacts and is never printed.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
import csv
import hashlib
import json
import os
import re
import time
import urllib.request

root = Path.home() / 'rsg-former'
dest = Path.home() / 'shared-assets/datasets/openimages_test'
annotations = dest / 'annotations'
annotations.mkdir(parents=True, exist_ok=True)
evaluation = dest / 'evaluation_only'
evaluation.mkdir(exist_ok=True)
report = root / 'reports/OPENIMAGES_TEST_METADATA.json'
sources = {
    'test-images-with-rotation.csv': 'https://storage.googleapis.com/openimages/2018_04/test/test-images-with-rotation.csv',
    'oidv6-test-annotations-vrd.csv': 'https://storage.googleapis.com/openimages/v6/oidv6-test-annotations-vrd.csv',
    'open_images_test_captions.jsonl': 'https://storage.googleapis.com/localized-narratives/annotations/open_images_test_captions.jsonl'
}
started = time.monotonic()
state = {'schema_version': 1, 'scope': 'Open Images TEST only; no train/validation',
         'mode': 'metadata_human_reference_lock_no_pixels_no_performance_scoring',
         'reference_policy': 'evaluation-only; do not use caption text for prompts, calibration, selection, or training',
         'assets': {}}

def emit(status, **values):
    state.update(status=status, updated_utc=datetime.now(timezone.utc).isoformat(), **values)
    report.write_text(json.dumps(state, indent=2))

def guard():
    assert time.monotonic() - started < 600, 'metadata preparation wall cap'
    assert os.statvfs(root).f_bavail * os.statvfs(root).f_frsize >= 16 * 2**30, 'disk reserve'

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()

def flickr_identity(url):
    parsed = urlparse(url)
    host = parsed.hostname or ''
    if host.endswith('staticflickr.com') or host.endswith('static.flickr.com'):
        key = Path(parsed.path).stem.split('_')[0]
        return key if key.isdigit() else None
    if host == 'flickr.com' or host.endswith('.flickr.com'):
        found = re.search(r'/photos/[^/]+/(\d+)', parsed.path)
        return found.group(1) if found else None
    return None

try:
    if report.exists() and json.loads(report.read_text()).get('status') == 'failed':
        preserved = root / 'reports/OPENIMAGES_TEST_METADATA_INITIAL_FAILURE.json'
        if not preserved.exists(): preserved.write_text(report.read_text())
    for name, url in sources.items():
        guard(); emit('downloading', current_asset=name)
        path = (evaluation if 'captions' in name else annotations) / name
        legacy = annotations / name
        if 'captions' in name and legacy.is_file() and not path.exists():
            assert dest.resolve() in legacy.resolve().parents and dest.resolve() in path.resolve().parents
            legacy.rename(path)  # Single owned evaluation asset, no deletion/overwrite.
        if not path.exists():
            partial = path.with_suffix(path.suffix + '.part')
            with urllib.request.urlopen(url, timeout=30) as response, partial.open('wb') as stream:
                length = int(response.headers.get('Content-Length', '0'))
                assert 0 < length < 256 * 2**20, 'unexpected individual metadata size'
                count = 0
                while True:
                    guard(); block = response.read(1024 * 1024)
                    if not block: break
                    stream.write(block); count += len(block)
                    assert count < 256 * 2**20
                assert count == length, 'incomplete metadata download'
            partial.replace(path)
        state['assets'][name] = {'url': url, 'bytes': path.stat().st_size, 'sha256': digest(path),
                                'relative_path': str(path.relative_to(dest))}
    assert sum(a['bytes'] for a in state['assets'].values()) < 512 * 2**20
    emit('validating')
    metadata = {}
    flickr_to_oi = defaultdict(list)
    with (annotations / 'test-images-with-rotation.csv').open(encoding='utf8', newline='') as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames
        for row in reader:
            identity = row['ImageID']
            assert identity not in metadata and re.fullmatch('[0-9a-f]{16}', identity)
            record = {key: row.get(key) for key in ['ImageID', 'OriginalURL', 'OriginalMD5', 'Rotation', 'License']}
            metadata[identity] = record
            key = flickr_identity(row.get('OriginalURL', ''))
            if key: flickr_to_oi[key].append(identity)
    refs = defaultdict(list)
    annotation_count = 0
    dataset_labels = Counter()
    with (evaluation / 'open_images_test_captions.jsonl').open(encoding='utf8') as stream:
        for line in stream:
            guard(); item = json.loads(line)
            identity = item['image_id']
            assert identity in metadata, 'caption identity outside protected OI TEST'
            assert isinstance(item['caption'], str) and item['caption'].strip()
            refs[identity].append(item['caption'])
            dataset_labels[item['dataset_id']] += 1
            annotation_count += 1
    vrd_ids = set(); non_attribute_vrd_ids = set(); relation_labels = Counter(); vrd_count = 0
    with (annotations / 'oidv6-test-annotations-vrd.csv').open(encoding='utf8', newline='') as stream:
        reader = csv.DictReader(stream)
        assert 'RelationshipLabel' in reader.fieldnames
        for row in reader:
            assert row['ImageID'] in metadata
            vrd_ids.add(row['ImageID']); relation_labels[row['RelationshipLabel']] += 1; vrd_count += 1
            if row['RelationshipLabel'] != 'is': non_attribute_vrd_ids.add(row['ImageID'])
    ordered = sorted(refs)
    mapping = {identity: index + 1 for index, identity in enumerate(ordered)}
    coco = {'info': {'description': 'Official human Localized Narratives OI test captions; evaluation-only'},
            'images': [{'id': mapping[identity], 'openimages_id': identity} for identity in ordered],
            'annotations': []}
    for identity in ordered:
        for caption in refs[identity]:
            coco['annotations'].append({'id': len(coco['annotations']) + 1,
                                        'image_id': mapping[identity], 'caption': caption})
    reference_path = evaluation / 'test_references_coco.json'
    reference_path.write_text(json.dumps(coco, ensure_ascii=False, separators=(',', ':')))
    manifest_path = dest / 'test_identity_manifest.json'
    manifest_path.write_text(json.dumps({'scope': 'test_only', 'images': [metadata[identity] for identity in sorted(metadata)],
                                         'caption_numeric_id_by_oi_id': mapping,
                                         'caption_test_ids': ordered, 'vrd_test_ids': sorted(vrd_ids),
                                         'reference_text_included': False}, separators=(',', ':')))
    # Join original Flickr identities against VG metadata; no caption contents used.
    vg = json.loads((root / 'data/vg150/image_data.json').read_text())
    matches = []
    for image in vg:
        key = str(image.get('flickr_id') or '')
        if key in flickr_to_oi:
            matches.append({'vg_image_id': image['image_id'], 'flickr_id': key,
                            'oi_test_ids': flickr_to_oi[key]})
    match_path = root / 'artifacts/caption_readiness/oi_known_identity_matches.json'
    match_path.parent.mkdir(parents=True, exist_ok=True)
    match_path.write_text(json.dumps({'evidence': 'VG flickr_id equals OI test OriginalURL identity',
                                      'matches': matches, 'reference_text_included': False}, separators=(',', ':')))
    emit('complete', test_images=len(metadata), caption_images=len(refs), human_captions=annotation_count,
         caption_dataset_ids=dict(dataset_labels), vrd_images=len(vrd_ids), vrd_rows=vrd_count,
         vrd_non_attribute_images=len(non_attribute_vrd_ids), vrd_non_attribute_rows=vrd_count-relation_labels['is'],
         vrd_relation_counts=dict(relation_labels), captions_missing_for_test_images=len(set(metadata) - set(refs)),
         metadata_columns=columns, flickr_identity_coverage=len(flickr_to_oi),
         vg_known_oi_test_identity_overlap=len(matches),
         identity_manifest_sha256=digest(manifest_path), reference_sha256=digest(reference_path),
         overlap_manifest_sha256=digest(match_path), images_downloaded_by_this_task=0,
         full_test_pixels_ready=False, full_clean_release_certified=False, test_scored=False)
    print(json.dumps(state, indent=2))
except Exception as error:
    emit('failed', error_type=type(error).__name__, reason=str(error)[:240])
    print(json.dumps(state, indent=2)); raise
