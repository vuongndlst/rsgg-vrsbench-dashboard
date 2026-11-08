"""Merge known held-out image identities; never calls the result fully clean."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

root = Path.home() / 'rsg-former'
folder = root / 'artifacts/caption_readiness'
old = json.loads((folder / 'known_identity_exclusions.json').read_text())
oi = json.loads((folder / 'oi_known_identity_matches.json').read_text())
meta = json.loads((root / 'reports/OPENIMAGES_TEST_METADATA.json').read_text())
assert meta['status'] == 'complete' and meta['test_scored'] is False
protected = set(old['excluded_known_vg_train_val_ids'])
oi_ids = {int(match['vg_image_id']) for match in oi['matches']}
full = {}
for split in ('train', 'val'):
    restored = set(old['provisional_split_image_ids'][split]) | {
        match['vg_image_id'] for match in old['identity_matches'] if match['vg_split'] == split}
    assert len(restored) == (57723 if split == 'train' else 5000)
    full[split] = restored
combined = protected | (oi_ids & (full['train'] | full['val']))
splits = {split: sorted(ids - combined) for split, ids in full.items()}
report = {'schema_version': 1, 'updated_utc': datetime.now(timezone.utc).isoformat(),
          'status': 'PROVISIONAL_KNOWN_IDENTITY_EXCLUSIONS_NOT_FULL_PIXEL_CLEAN_RELEASE',
          'oi_known_vg_overlap_all_metadata_images': len(oi_ids),
          'oi_vg_train_matches': len(oi_ids & full['train']),
          'oi_vg_val_matches': len(oi_ids & full['val']),
          'new_train_val_exclusions_after_coco_flickr': len(combined-protected),
          'combined_excluded_train_val_images': len(combined),
          'provisional_sorted_split_counts': {split: len(ids) for split, ids in splits.items()},
          'provisional_sorted_split_sha256': {split: hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest() for split, ids in splits.items()},
          'oi_test_identity_source_sha256': meta['assets']['test-images-with-rotation.csv']['sha256'],
          'reference_text_used_for_exclusions': False, 'fresh_training_launched': False,
          'remaining': ['exact/perceptual duplicates and unresolved identity mappings',
                        'fresh initialization provenance and resource/augmentation/evaluator smoke',
                        'full protected image manifests with a versioned training preregistration']}
manifest = dict(report, split_image_ids=splits, excluded_known_image_ids=sorted(combined),
                oi_identity_matches=oi['matches'])
target = folder / 'canonical_known_identity_clean_splits.json'
target.write_text(json.dumps(manifest, separators=(',', ':')))
report['manifest_sha256'] = hashlib.sha256(target.read_bytes()).hexdigest()
text = json.dumps(report, indent=2)
(root / 'reports/CAPTION_CLEAN_IDENTITY_MERGE.json').write_text(text)
print(text)
