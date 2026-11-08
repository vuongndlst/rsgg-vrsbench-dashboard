"""Independent metadata-only audit; never reads images, scores, or model weights.

Writes readiness artifacts outside locked v2 source/config/feature directories.
Identity matches are a lower bound, not a substitute for an image duplicate audit.
"""
from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import urlparse
import hashlib
import json
import os
import sys
import numpy as np

home = Path.home()
root = home / 'rsg-former'
sys.path[:0] = [str(root / 'src'), str(root / '.tools/deps')]
from rsgformer.data import VG150

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def id_sha(ids):
    return hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()

registry = json.loads((home / 'shared-assets/ASSET_REGISTRY.json').read_text())
source = {}
protected_coco = {}
protected_flickr = {}
for name, filename in [('coco_karpathy', 'dataset_coco.json'),
                       ('flickr30k_karpathy', 'dataset_flickr30k.json')]:
    folder = home / 'shared-assets/datasets' / name
    path = folder / 'annotations' / filename
    actual_sha = sha(path)
    expected_sha = registry['datasets'][name]['metadata_sha256']
    assert actual_sha == expected_sha, 'Shared caption metadata changed'
    raw = json.loads(path.read_text())
    counts = {}
    protected = protected_coco if name == 'coco_karpathy' else protected_flickr
    for item in raw['images']:
        split = item['split']
        counts[split] = counts.get(split, 0) + 1
        if split not in ('test', 'val'):
            continue
        # Image identities only: no sentences/reference text in training manifests.
        if name == 'coco_karpathy':
            key = int(item.get('cocoid') or Path(item['filename']).stem.split('_')[-1])
        else:
            key = Path(item['filename']).stem.split('_')[0]
        assert key not in protected, 'Duplicate protected identity'
        protected[key] = split
    source[name] = {'metadata_sha256': actual_sha, 'split_counts': counts,
                    'test_image_coverage': registry['datasets'][name]['available_test_images'],
                    'expected_test_images': registry['datasets'][name]['expected_test_images']}

ds = VG150(root / 'data/vg150', home / 'sbir/hienlth/vg', 'train', train_overlap=False)
arr = ds.arrays
eligible = (arr['img_to_first_box'] >= 0) & (arr['img_to_first_rel'] >= 0)
regular = np.flatnonzero((arr['split'] == 0) & eligible).tolist()
testing = np.flatnonzero((arr['split'] == 2) & eligible).tolist()
split_rows = {'train': regular[5000:], 'val': regular[:5000], 'test': testing}
assert split_rows['train'] == ds.indices
matches = []
counts = {s: {'images': len(rows), 'coco_test': 0, 'coco_val': 0,
              'flickr_test': 0, 'flickr_val': 0} for s, rows in split_rows.items()}
excluded = set()
flickr_identity_fields = 0
for vg_split, rows in split_rows.items():
    for row in rows:
        item = ds.image_info[row]
        coco_id = item.get('coco_id')
        flickr_id = item.get('flickr_id')
        if flickr_id:
            flickr_identity_fields += 1
        found = []
        if coco_id and int(coco_id) in protected_coco:
            found.append(('coco', protected_coco[int(coco_id)], str(int(coco_id)), 'coco_id'))
        if flickr_id and str(flickr_id) in protected_flickr:
            found.append(('flickr', protected_flickr[str(flickr_id)], str(flickr_id), 'flickr_id'))
        # Only use Flickr-hosted URL identities, never numeric VG filenames.
        for key in ['url', 'flickr_url']:
            url = item.get(key, '') or ''
            host = urlparse(url).hostname or ''
            if host.endswith('flickr.com') or host.endswith('staticflickr.com'):
                identity = Path(urlparse(url).path).stem.split('_')[0]
                if identity in protected_flickr and not any(m[0] == 'flickr' for m in found):
                    found.append(('flickr', protected_flickr[identity], identity, key))
        for dataset, split, identity, evidence in found:
            counts[vg_split][dataset + '_' + split] += 1
            matches.append({'vg_image_id': item['image_id'], 'vg_split': vg_split,
                            'external_dataset': dataset, 'external_split': split,
                            'external_identity': identity, 'evidence': evidence})
            if vg_split in ('train', 'val'):
                excluded.add(item['image_id'])

vg_raw = home / 'sbir/hienlth/vg/region_descriptions.json'
region_error = None
repair_path = root / 'reports/VG_REGION_REFERENCE_PREPARATION.json'
repair = json.loads(repair_path.read_text()) if repair_path.exists() else {}
if repair.get('status') == 'complete':
    # Already streaming-validated by preparation. Recheck the content digest;
    # do not load the 712MB reference text into this metadata-only audit.
    vg_raw = root / 'data/vg_caption/region_descriptions.json'
    assert sha(vg_raw) == repair['json_sha256'], 'Prepared reference content changed'
    region_ids = json.loads((root / 'data/vg_caption/region_image_ids.json').read_text())
    assert len(region_ids) == repair['image_records']
    region_total = repair['region_count']
    nonempty = repair['nonempty_phrases']
    region_fields = None  # Full schema contracts belong to the crop evaluator audit.
else:
    try:
        region_data = json.loads(vg_raw.read_text())
    except json.JSONDecodeError as error:
        region_data = []
        region_error = {'error': type(error).__name__, 'message': error.msg, 'position': error.pos,
                        'status': 'INVALID_JSON_DO_NOT_USE_AS_HUMAN_REFERENCES'}
    assert isinstance(region_data, list)
    region_ids = [int(item.get('id', item.get('image_id'))) for item in region_data]
    region_total = sum(len(item.get('regions', [])) for item in region_data) if not region_error else None
    nonempty = sum(bool(region.get('phrase', '').strip()) for item in region_data for region in item.get('regions', [])) if not region_error else None
    region_fields = sorted({key for item in region_data[:20] for region in item.get('regions', []) for key in region}) if not region_error else None
assert len(set(region_ids)) == len(region_ids), 'Duplicate region image identity'
region_set = set(region_ids)
candidate = {split: [ds.image_info[row]['image_id'] for row in split_rows[split]
                      if ds.image_info[row]['image_id'] not in excluded]
             for split in ('train', 'val')}
out = {'schema_version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
       'mode': 'metadata_only_no_image_decode_no_model_no_performance_scoring',
       'source_metadata': source,
       'vg_image_metadata_sha256': sha(root / 'data/vg150/image_data.json'),
       'v2_split_hashes': {s: id_sha([ds.image_info[row]['image_id'] for row in rows]) for s, rows in split_rows.items()},
       'identity_overlap': counts, 'matched_records': len(matches),
       'vg_train_val_known_external_val_test_exclusions': len(excluded),
       'flickr_identity_field_rows': flickr_identity_fields,
       'vg_region_references': {'bytes': vg_raw.stat().st_size, 'sha256': sha(vg_raw),
                               'parse_error': region_error,
                               'source': repair.get('source_url') if repair.get('status') == 'complete' else 'existing_original_asset',
                               'invalid_geometry_records': repair.get('invalid_geometry_records'),
                               'image_records': len(region_ids), 'regions': region_total,
                               'nonempty_phrases': nonempty, 'fields': region_fields,
                               'split_image_coverage': {s: sum(ds.image_info[row]['image_id'] in region_set for row in rows) for s, rows in split_rows.items()},
                               'scope': 'human region phrases require matched crop evaluation; not global image references'},
       'provisional_clean_split_counts': {s: len(ids) for s, ids in candidate.items()},
       'status': 'PROVISIONAL_METADATA_EXCLUSIONS_NOT_CAPTION_CLEAN_RELEASE',
       'remaining_checks': ['COCO/Flickr/OI exact and perceptual image duplicate audit including VG images with absent identity fields',
                            'Open Images test identity/duplicate manifest and human reference release lock',
                            'canonical RelTR annotation/vocabulary/split parity',
                            'audited fresh backbone initialization and fresh detector/SG training',
                            'development images and complete Flickr test coverage'],
       'disk_available_bytes': os.statvfs(root).f_bavail * os.statvfs(root).f_frsize}
dest = root / 'artifacts/caption_readiness'
dest.mkdir(parents=True, exist_ok=True)
manifest = {'status': out['status'], 'excluded_known_vg_train_val_ids': sorted(excluded),
            'provisional_split_image_ids': candidate, 'identity_matches': matches,
            'source_metadata_sha256': {name: value['metadata_sha256'] for name, value in source.items()},
            'reference_text_included': False}
(dest / 'known_identity_exclusions.json').write_text(json.dumps(manifest, separators=(',', ':')))
out['identity_manifest_sha256'] = sha(dest / 'known_identity_exclusions.json')
text = json.dumps(out, indent=2)
(root / 'reports/VG_CAPTION_OVERLAP_AUDIT.json').write_text(text)
print(text)
