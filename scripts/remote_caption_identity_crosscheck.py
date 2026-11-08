"""Additional image-only identity crosschecks; no new train release or scoring."""
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone
from urllib.parse import urlparse
import hashlib
import json
import os
import re
import signal
import zipfile

signal.alarm(60)
os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
home, root = Path.home(), Path.home() / 'rsg-former'
meta_path = root / 'data/vg150/image_data.json'
raw_meta = meta_path.read_bytes()
metadata = json.loads(raw_meta)
by_id = {int(item['image_id']): item for item in metadata}
manifest_path = root / 'artifacts/caption_readiness/canonical_known_identity_clean_splits.json'
raw_manifest = manifest_path.read_bytes()
manifest = json.loads(raw_manifest)
archive = root / 'data/vg_canonical_audit/reltr_vg_annotations.zip'
with zipfile.ZipFile(archive) as bundle:
    splits = {split: {int(image['id']) for image in json.loads(bundle.read('vg/' + split + '.json'))['images']}
              for split in ['train', 'val', 'test']}

groups = defaultdict(list)
for item in metadata:
    if item.get('flickr_id'):
        groups[str(item['flickr_id'])].append(int(item['image_id']))
duplicates = {key: ids for key, ids in groups.items() if len(ids) > 1}
cross_split = []
for identity, ids in duplicates.items():
    roles = {split: sorted(set(ids) & members) for split, members in splits.items()}
    if sum(bool(value) for value in roles.values()) > 1:
        cross_split.append({'flickr_identity': identity, 'vg_ids_by_split': roles})

coco_source = home / 'image-captioning-kg/datasets/coco_val/annotations/captions_val2017.json'
source_raw = coco_source.read_bytes()
# Only export images[].id/flickr_url. Captions/annotations are never passed to any model/prompt.
coco_images = json.loads(source_raw)['images']
karpathy = home / 'shared-assets/datasets/coco_karpathy/annotations/dataset_coco.json'
raw_karpathy = karpathy.read_bytes()
assert hashlib.sha256(raw_karpathy).hexdigest() == '2fd999220673258012acfb411a4e7e66af7d488050b2519b0badcc49b7600b8d'
protected = {int(item['cocoid']): item['split'] for item in json.loads(raw_karpathy)['images']
             if item['split'] in ['test', 'val']}
protected_flickr = {}
covered = set()
for image in coco_images:
    identity = int(image['id'])
    if identity not in protected:
        continue
    covered.add(identity)
    url = image.get('flickr_url') or ''
    parsed = urlparse(url)
    if (parsed.hostname or '').endswith(('flickr.com', 'staticflickr.com')):
        flickr_id = Path(parsed.path).stem.split('_')[0]
        if flickr_id.isdigit():
            protected_flickr.setdefault(flickr_id, []).append({'coco_id': identity, 'split': protected[identity]})

new_matches = []
for split, identities in manifest['split_image_ids'].items():
    for vg_id in identities:
        item = by_id[int(vg_id)]
        matches = protected_flickr.get(str(item.get('flickr_id')), [])
        for match in matches:
            new_matches.append({'vg_image_id': int(vg_id), 'vg_split': split,
                                'external_dataset': 'coco_karpathy', 'external_split': match['split'],
                                'coco_id': match['coco_id'], 'evidence': 'existing_COCO_val2017_images_flickr_url'})
artifact = root / 'artifacts/caption_readiness/additional_identity_crosscheck.json'
artifact.write_text(json.dumps({'new_known_identity_matches': new_matches,
                                'vg_cross_split_flickr_identity_matches': cross_split}, separators=(',', ':')))
report = {
    'status': 'IMAGE_IDENTITY_CROSSCHECK_COMPLETE', 'utc': datetime.now(timezone.utc).isoformat(),
    'source_sha256': hashlib.sha256(source_raw).hexdigest(), 'source_filename': coco_source.name,
    'source_images': len(coco_images), 'protected_coco_ids': len(protected),
    'protected_coco_ids_covered_by_existing_source': len(covered),
    'protected_flickr_identities_recovered': len(protected_flickr),
    'new_matches_after_prior_exclusions': len(new_matches),
    'new_vg_train_ids': len({match['vg_image_id'] for match in new_matches if match['vg_split'] == 'train'}),
    'new_vg_val_ids': len({match['vg_image_id'] for match in new_matches if match['vg_split'] == 'val'}),
    'vg_metadata_duplicate_flickr_identity_groups': len(duplicates),
    'vg_cross_canonical_split_flickr_identity_groups': len(cross_split),
    'evidence_artifact_sha256': hashlib.sha256(artifact.read_bytes()).hexdigest(),
    'provisional_manifest_sha256': hashlib.sha256(raw_manifest).hexdigest(),
    'reference_text_used_for_exclusions': False, 'source_contains_caption_text': True,
    'only_image_identifier_fields_used': True, 'image_pixels_read': False,
    'training_manifest_modified': False, 'live_v2_modified': False, 'benchmark_test_scored': False,
    'remaining': ['Full COCO protected image-to-Flickr identity source coverage',
                  'Exact/perceptual pixel duplicate audit, including validation identities missing metadata',
                  'Merge any verified extra exclusions in a separate versioned clean-release manifest before fresh training'],
}
(root / 'reports/CAPTION_IDENTITY_CROSSCHECK.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report))
