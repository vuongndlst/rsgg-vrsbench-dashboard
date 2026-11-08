"""Verify metadata-identified cross-split duplicates directly inside VG ZIPs."""
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import io
import itertools
import json
import os
import signal
import struct
import zipfile
import numpy as np
from PIL import Image

signal.alarm(90)
os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
home, root = Path.home(), Path.home() / 'rsg-former'
evidence_path = root / 'artifacts/caption_readiness/additional_identity_crosscheck.json'
evidence_bytes = evidence_path.read_bytes()
evidence = json.loads(evidence_bytes)
groups = evidence['vg_cross_split_flickr_identity_matches']
identities = sorted({identity for group in groups for ids in group['vg_ids_by_split'].values() for identity in ids})
assert len(identities) <= 1000
zips = [zipfile.ZipFile(home / 'sbir/hienlth/vg' / name) for name in ['images.zip', 'images2.zip']]
members = {}
for index, bundle in enumerate(zips):
    for info in bundle.infolist():
        if info.filename.lower().endswith('.jpg') and Path(info.filename).stem.isdigit():
            identity = int(Path(info.filename).stem)
            assert identity not in members
            members[identity] = (index, info)
records = {}
thumbnails = {}
read_bytes = 0
try:
    for identity in identities:
        index, info = members[identity]
        assert 0 < info.file_size <= 8 * 1024 ** 2
        read_bytes += info.file_size
        assert read_bytes <= 256 * 1024 ** 2
        data = zips[index].read(info)
        assert len(data) == info.file_size
        with Image.open(io.BytesIO(data)) as decoded:
            rgb = decoded.convert('RGB')
            assert rgb.width * rgb.height < 16_000_000
            shape = [rgb.width, rgb.height]
            digest = hashlib.sha256(struct.pack('>II', *shape) + rgb.tobytes()).hexdigest()
            thumbnails[identity] = np.asarray(rgb.resize((64, 64), Image.Resampling.BILINEAR), dtype=np.float32)
            gray = np.asarray(rgb.convert('L').resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float64)
            basis = np.cos(np.pi / 32 * np.arange(8)[:, None] * (np.arange(32)[None, :] + .5))
            basis *= np.sqrt(2 / 32)
            basis[0] /= np.sqrt(2)
            low = (basis @ gray @ basis.T).reshape(-1)[1:]
            bits = low > np.median(low)
            phash = sum(int(value) << index for index, value in enumerate(bits))
        records[identity] = {'vg_id': identity, 'jpeg_sha256': hashlib.sha256(data).hexdigest(),
                             'decoded_rgb_sha256': digest, 'size': shape, 'phash63': hex(phash)}
finally:
    for bundle in zips:
        bundle.close()

roles = [('train', 'val'), ('train', 'test'), ('val', 'test')]
summary = {a + '_' + b: {'known_same_flickr_identity_groups': 0, 'exact_jpeg_groups': 0,
                        'exact_decoded_rgb_groups': 0, 'exact_decoded_rgb_left_ids': set(),
                        'phash_le4_rgb_mae_le3_groups': 0, 'phash_le8_rgb_mae_le10_groups': 0} for a, b in roles}
provisional_path = root / 'artifacts/caption_readiness/canonical_known_identity_clean_splits.json'
provisional = json.loads(provisional_path.read_text())
candidate_train = set(provisional['split_image_ids']['train'])
remaining_provisional_train_alias_ids = set()
for group in groups:
    group['pixel_evidence'] = []
    for left, right in roles:
        left_ids, right_ids = group['vg_ids_by_split'][left], group['vg_ids_by_split'][right]
        if not left_ids or not right_ids:
            continue
        metric = summary[left + '_' + right]
        metric['known_same_flickr_identity_groups'] += 1
        byte_pairs, rgb_pairs, similarities = [], [], []
        for a, b in itertools.product(left_ids, right_ids):
            if records[a]['jpeg_sha256'] == records[b]['jpeg_sha256']:
                byte_pairs.append([a, b])
            if records[a]['decoded_rgb_sha256'] == records[b]['decoded_rgb_sha256']:
                rgb_pairs.append([a, b])
                metric['exact_decoded_rgb_left_ids'].add(a)
            hamming = (int(records[a]['phash63'], 16) ^ int(records[b]['phash63'], 16)).bit_count()
            mae = float(np.abs(thumbnails[a] - thumbnails[b]).mean())
            similarities.append({'vg_ids': [a, b], 'phash63_hamming': hamming, 'rgb64_mean_absolute_error': mae})
        metric['exact_jpeg_groups'] += bool(byte_pairs)
        metric['exact_decoded_rgb_groups'] += bool(rgb_pairs)
        metric['phash_le4_rgb_mae_le3_groups'] += any(pair['phash63_hamming'] <= 4 and pair['rgb64_mean_absolute_error'] <= 3 for pair in similarities)
        metric['phash_le8_rgb_mae_le10_groups'] += any(pair['phash63_hamming'] <= 8 and pair['rgb64_mean_absolute_error'] <= 10 for pair in similarities)
        group['pixel_evidence'].append({'split_pair': [left, right], 'same_jpeg_pairs': byte_pairs,
                                        'same_decoded_rgb_pairs': rgb_pairs, 'similarities': similarities})
        if left == 'train':
            remaining_provisional_train_alias_ids.update(set(left_ids) & candidate_train)
for metric in summary.values():
    metric['exact_decoded_rgb_left_ids'] = sorted(metric['exact_decoded_rgb_left_ids'])
    metric['exact_decoded_rgb_left_id_count'] = len(metric['exact_decoded_rgb_left_ids'])
artifact = root / 'artifacts/caption_readiness/vg_cross_split_pixel_evidence.json'
history = artifact.with_name('vg_cross_split_pixel_evidence_exact_only.json')
if artifact.exists() and not history.exists():
    history.write_bytes(artifact.read_bytes())
artifact.write_text(json.dumps({'groups': groups, 'records': list(records.values()),
                                'provisional_train_alias_ids_to_protect': sorted(remaining_provisional_train_alias_ids)},
                               separators=(',', ':')))
report = {
    'status': 'CROSS_SPLIT_PIXEL_AUDIT_COMPLETE', 'utc': datetime.now(timezone.utc).isoformat(),
    'scope': 'metadata-identified Flickr aliases only; not full pairwise/perceptual dataset audit',
    'identity_source_sha256': hashlib.sha256(evidence_bytes).hexdigest(),
    'groups_checked': len(groups), 'unique_images_decoded': len(records), 'zip_member_bytes_read': read_bytes,
    'split_pair_summary': summary,
    'remaining_provisional_train_alias_ids_to_protect': len(remaining_provisional_train_alias_ids),
    'pixel_evidence_sha256': hashlib.sha256(artifact.read_bytes()).hexdigest(),
    'orientation': 'raw decoded RGB, no EXIF transform; dimensions included in hash',
    'perceptual_threshold_scope': 'two diagnostic thresholds only; not a finalized full-release exclusion criterion',
    'vg_images_extracted': False, 'image_reference_captions_read': False,
    'training_manifest_modified': False, 'live_v2_modified': False,
    'benchmark_validation_scored': False, 'benchmark_test_scored': False,
    'remaining': ['Versioned clean-release exclusion of internal held-out aliases before fresh training',
                  'Full external protected identity/pixel/perceptual coverage and missing validation identities',
                  'Preserve running v2 protocol and report duplicate-split limitation; never relabel it caption-clean'],
}
(root / 'reports/VG_CROSS_SPLIT_PIXEL_AUDIT.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report))
