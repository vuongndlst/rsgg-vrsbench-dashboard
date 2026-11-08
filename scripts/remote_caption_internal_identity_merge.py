"""Versioned draft protecting internal VG held-out photo identities as well as external IDs."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

root = Path.home() / 'rsg-former'
folder = root / 'artifacts/caption_readiness'
previous_path = folder / 'canonical_known_identity_clean_splits.json'
previous_bytes = previous_path.read_bytes()
assert hashlib.sha256(previous_bytes).hexdigest() == '9f23205cb05ac6689253e069ecc37b0f496cd313c88d99a5619332f5ca999485'
previous = json.loads(previous_bytes)
cross_path = folder / 'additional_identity_crosscheck.json'
cross_bytes = cross_path.read_bytes()
cross = json.loads(cross_bytes)
removed = {'train': set(), 'val': set()}
for group in cross['vg_cross_split_flickr_identity_matches']:
    roles = group['vg_ids_by_split']
    if roles['val'] or roles['test']:
        removed['train'].update(roles['train'])
    if roles['test']:
        removed['val'].update(roles['val'])
splits = {split: sorted(set(ids) - removed[split]) for split, ids in previous['split_image_ids'].items()}
actual_removed = {split: sorted(set(previous['split_image_ids'][split]) - set(splits[split])) for split in splits}
assert len(actual_removed['train']) == 143
manifest = {
    'schema_version': 2,
    'status': 'PROVISIONAL_EXTERNAL_AND_INTERNAL_IDENTITIES_NOT_FULL_PIXEL_CLEAN_RELEASE',
    'updated_utc': datetime.now(timezone.utc).isoformat(),
    'previous_manifest_sha256': hashlib.sha256(previous_bytes).hexdigest(),
    'cross_identity_evidence_sha256': hashlib.sha256(cross_bytes).hexdigest(),
    'exclusion_rule': 'metadata Flickr photo identity shared with canonical VG val/test removes train; identity shared with VG test removes val',
    'perceptual_threshold_used_for_exclusions': False,
    'split_image_ids': splits, 'new_internal_alias_excluded_ids': actual_removed,
    'excluded_known_image_ids': sorted(set(previous['excluded_known_image_ids']) | set().union(*removed.values())),
    'reference_text_or_test_scores_used': False, 'fresh_training_launched': False,
    'clean_release_certified': False,
}
target = folder / 'canonical_internal_external_identity_splits_v2.json'
assert not target.exists(), 'preserve_existing_versioned_manifest'
target.write_text(json.dumps(manifest, separators=(',', ':')))
report = {key: value for key, value in manifest.items() if key not in ['split_image_ids', 'excluded_known_image_ids', 'new_internal_alias_excluded_ids']}
report.update(manifest_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
              manifest_alias='${REMOTE_HOME}/rsg-former/artifacts/caption_readiness/' + target.name,
              new_internal_exclusion_counts={split: len(ids) for split, ids in actual_removed.items()},
              provisional_split_counts={split: len(ids) for split, ids in splits.items()},
              provisional_split_sha256={split: hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest() for split, ids in splits.items()},
              previous_manifest_preserved=True, live_v2_modified=False, benchmark_test_scored=False,
              draft_prior_for_previous_manifest_is_current=False,
              remaining=['Full external protected identity/pixel/perceptual audit and unresolved validation identities',
                         'Regenerate train-only prior from final release; old draft never used in model',
                         'Resource smoke and separate source/split/selection preregistration before fresh training'])
(root / 'reports/CAPTION_INTERNAL_IDENTITY_MERGE.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report))
