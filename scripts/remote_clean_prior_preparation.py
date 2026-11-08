"""Reconstruct prior indexing on canonical TRAIN; prepare provisional clean counts.

No validation/test annotations or predictions, no edits to the live experiment.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import signal
import sys
import time
import numpy as np
import torch
torch.set_num_threads(1)
signal.alarm(180)
root = Path.home() / 'rsg-former'
sys.path.insert(0, str(root / 'src'))
from rsgformer.vg_canonical_data import VGCanonical
started = time.monotonic()
manifest_path = root / 'artifacts/caption_readiness/canonical_known_identity_clean_splits.json'
manifest = json.loads(manifest_path.read_text())
expected_manifest_sha = json.loads((root / 'reports/CAPTION_CLEAN_IDENTITY_MERGE.json').read_text())['manifest_sha256']
assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == expected_manifest_sha
assert manifest['status'].startswith('PROVISIONAL_')
ds = VGCanonical(root / 'data/vg_canonical_audit/reltr_vg_annotations.zip', Path.home() / 'sbir/hienlth/vg', 'train')
author_path = next((root / 'vendor/salience_sgg').rglob('vg_prior_static.pt'))
assert hashlib.sha256(author_path.read_bytes()).hexdigest() == 'f4c84b9c0ad294aefbc10bb6112a0f4a6008d80e3203e8a34a6309c150421bcf'
author = torch.load(author_path, map_location='cpu', weights_only=False)
assert isinstance(author, np.ndarray) and author.shape == (151, 151, 50)
full_zero = np.zeros_like(author)
full_one = np.zeros_like(author)
clean_zero = np.zeros_like(author)
allowed = set(manifest['split_image_ids']['train'])
assert allowed == set(ds.ids) - set(manifest['excluded_known_image_ids'])
full_relations = clean_relations = 0
for identity in ds.ids:
    labels = np.array([annotation['category_id'] for annotation in ds.annotations[identity]], dtype=np.int64)
    relations = np.array(ds.relations[str(identity)], dtype=np.int64).reshape(-1, 3)
    assert len(relations) == len(set(map(tuple, relations)))
    subjects = labels[relations[:, 0]]
    objects = labels[relations[:, 1]]
    predicates = relations[:, 2] - 1
    assert (subjects >= 1).all() and (objects >= 1).all() and (predicates >= 0).all()
    np.add.at(full_one, (subjects, objects, predicates), 1)
    np.add.at(full_zero, (subjects - 1, objects - 1, predicates), 1)
    full_relations += len(relations)
    if identity in allowed:
        np.add.at(clean_zero, (subjects - 1, objects - 1, predicates), 1)
        clean_relations += len(relations)
assert full_relations == 315642 and int(clean_zero.sum()) == clean_relations
zero_equal = bool(np.array_equal(full_zero, author))
one_equal = bool(np.array_equal(full_one, author))
assert zero_equal != one_equal, 'author_prior_count_construction_not_reconstructed_exactly'
# Save a draft with explicit zero-based model label indexing. It must not be used
# until the final exclusion release is locked and the source protocol is registered.
folder = root / 'artifacts/caption_readiness'
target = folder / 'provisional_clean_train_prior_zero_based.pt'
torch.save(clean_zero, target)
out = {'status': 'PROVISIONAL_TRAIN_PRIOR_PREPARED', 'utc': datetime.now(timezone.utc).isoformat(),
       'elapsed_seconds': time.monotonic() - started, 'canonical_train_images': len(ds),
       'canonical_unique_relations': full_relations,
       'author_prior_exactly_equals_zero_based_counts': zero_equal,
       'author_prior_exactly_equals_one_based_counts': one_equal,
       'model_logits_class_indexing': 'zero-based 0..149',
       'draft_prior_object_indexing': 'zero-based 0..149; unused padding index150',
       'draft_prior_predicate_indexing': 'zero-based 0..49; no background channel',
       'draft_shape': list(clean_zero.shape), 'draft_dtype': str(clean_zero.dtype),
       'provisional_train_images': len(allowed), 'provisional_train_relations': clean_relations,
       'provisional_train_ids_sha256': manifest['provisional_sorted_split_sha256']['train'],
       'provisional_exclusion_manifest_sha256': expected_manifest_sha,
       'draft_prior_path': str(target.relative_to(root)),
       'draft_prior_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
       'draft_prior_bytes': target.stat().st_size,
       'used_in_any_model': False, 'clean_release_certified': False,
       'cuda_used': False, 'training_started': False, 'test_scored': False,
       'remaining': ['Finalize image overlap/exclusion release and regenerate prior',
                     'Register explicit prior indexing/normalization protocol before fresh SGG',
                     'Do not alter prior or checkpoint buffers in live v2']}
text = json.dumps(out, indent=2)
(root / 'reports/CLEAN_TRAIN_PRIOR_PREPARATION.json').write_text(text)
print(text)
