"""Compare annotation metadata with the official release; no images or scoring."""
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict
import hashlib
import json
import os
import sys
import zipfile
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np

root = Path.home() / 'rsg-former'
sys.path[:0] = [str(root / 'src'), str(root / '.tools/deps')]
from rsgformer.data import VG150

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()

def order_hash(ids):
    return hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()

archive = root / 'data/vg_canonical_audit/reltr_vg_annotations.zip'
prep = json.loads((root / 'reports/VG_CANONICAL_ANNOTATION_PREPARATION.json').read_text())
assert prep['status'] == 'download_complete_parity_pending'
assert sha(archive) == prep['archive_sha256']
ds = VG150(root / 'data/vg150', Path.home() / 'sbir/hienlth/vg', 'train', train_overlap=False)
arr = ds.arrays
eligible = (arr['img_to_first_box'] >= 0) & (arr['img_to_first_rel'] >= 0)
regular = np.flatnonzero((arr['split'] == 0) & eligible).tolist()
splits = {'train': regular[5000:], 'val': regular[:5000],
          'test': np.flatnonzero((arr['split'] == 2) & eligible).tolist()}
out = {'created_utc': datetime.now(timezone.utc).isoformat(),
       'mode': 'full_annotation_metadata_only_no_images_no_performance_scoring',
       'archive_sha256': prep['archive_sha256'], 'active_release_modified': False,
       'member_sha256': {}, 'splits': {}}
with zipfile.ZipFile(archive) as zipped:
    raw = zipped.read('vg/rel.json')
    out['member_sha256']['rel.json'] = hashlib.sha256(raw).hexdigest()
    relations = json.loads(raw)
    del raw
    predicate_names = relations['rel_categories']
    if isinstance(predicate_names, dict): predicate_names = {int(k): v for k, v in predicate_names.items()}
    else: predicate_names = dict(enumerate(predicate_names))
    out['predicate_vocabulary_mismatches'] = [i for i, name in ds.predicate_names.items() if predicate_names.get(i) != name]
    for split, rows in splits.items():
        raw = zipped.read('vg/' + split + '.json')
        out['member_sha256'][split + '.json'] = hashlib.sha256(raw).hexdigest()
        coco = json.loads(raw)
        del raw
        images = coco['images']
        ids = [int(image['id']) for image in images]
        assert len(ids) == len(set(ids))
        own_ids = [ds.image_info[row]['image_id'] for row in rows]
        own_rows = {ds.image_info[row]['image_id']: row for row in rows}
        canonical_ids = set(ids)
        own_set = set(own_ids)
        vocabulary = {int(category['id']): category['name'] for category in coco['categories']}
        annotations = defaultdict(list)
        for ann in coco['annotations']: annotations[int(ann['image_id'])].append(ann)
        summary = {'canonical_images': len(ids), 'adapter_images': len(own_ids),
                   'canonical_ordered_ids_sha256': order_hash(ids),
                   'adapter_ordered_ids_sha256': order_hash(own_ids),
                   'ordered_ids_equal': ids == own_ids, 'id_sets_equal': canonical_ids == own_set,
                   'canonical_only_count': len(canonical_ids - own_set),
                   'adapter_only_count': len(own_set - canonical_ids),
                   'canonical_only_sample': sorted(canonical_ids - own_set)[:20],
                   'adapter_only_sample': sorted(own_set - canonical_ids)[:20],
                   'object_vocabulary_mismatches': [i for i, name in ds.object_names.items() if vocabulary.get(i) != name],
                   'object_count_mismatch_images': 0, 'ordered_object_labels_mismatch_images': 0,
                   'dimension_mismatch_images': 0, 'ordered_raw_boxes_mismatch_images': 0,
                   'ordered_adapter_clipped_boxes_mismatch_images': 0,
                   'ordered_native_coco_clipped_boxes_mismatch_images': 0,
                   'relation_multiset_mismatch_images': 0,
                   'unique_relation_set_mismatch_images': 0,
                   'integer_inclusive_xywh_metadata_mismatch_images': 0,
                   'adapter_relations': 0, 'adapter_unique_relations': 0,
                   'canonical_annotation_objects': len(coco['annotations']),
                   'canonical_relations': sum(len(r) for r in relations[split].values()),
                   'max_raw_box_abs_error': 0.0, 'examples': []}
        for image in images:
            identity = int(image['id'])
            if identity not in own_rows: continue
            row = own_rows[identity]
            meta = ds.image_info[row]
            summary['dimension_mismatch_images'] += int((meta['width'], meta['height']) != (image['width'], image['height']))
            sample = ds._sample(row, clip=False)
            anns = annotations[identity]
            differences = []
            if len(anns) != len(sample.labels):
                summary['object_count_mismatch_images'] += 1
                differences.append('object_count')
            else:
                labels = np.array([ann['category_id'] for ann in anns])
                if not np.array_equal(labels, sample.labels):
                    summary['ordered_object_labels_mismatch_images'] += 1
                    differences.append('ordered_labels')
                boxes = np.array([ann['bbox'] for ann in anns], dtype=np.float64)
                projected = np.concatenate([np.floor(sample.boxes[:, :2]),
                                            np.floor(sample.boxes[:, 2:] - sample.boxes[:, :2]) + 1], axis=1)
                summary['integer_inclusive_xywh_metadata_mismatch_images'] += int(not np.array_equal(projected, boxes))
                boxes[:, 2:] += boxes[:, :2]
                error = float(np.abs(boxes - sample.boxes).max()) if len(boxes) else 0.0
                summary['max_raw_box_abs_error'] = max(summary['max_raw_box_abs_error'], error)
                if error > 0.001:
                    summary['ordered_raw_boxes_mismatch_images'] += 1
                    differences.append('raw_boxes')
                clipped_adapter = sample.boxes.copy()
                clipped_adapter[:, [0, 2]] = np.clip(clipped_adapter[:, [0, 2]], 0, meta['width'] - 1)
                clipped_adapter[:, [1, 3]] = np.clip(clipped_adapter[:, [1, 3]], 0, meta['height'] - 1)
                summary['ordered_adapter_clipped_boxes_mismatch_images'] += int(not np.allclose(boxes, clipped_adapter, atol=.001, rtol=0))
                native_clipped = boxes.copy()
                native_clipped[:, [0, 2]] = np.clip(native_clipped[:, [0, 2]], 0, meta['width'])
                native_clipped[:, [1, 3]] = np.clip(native_clipped[:, [1, 3]], 0, meta['height'])
                summary['ordered_native_coco_clipped_boxes_mismatch_images'] += int(not np.allclose(native_clipped, clipped_adapter, atol=.001, rtol=0))
            canonical_rel = sorted(tuple(int(v) for v in rel) for rel in relations[split].get(str(identity), []))
            own_rel = sorted(tuple(int(v) for v in rel) for rel in sample.relations)
            summary['adapter_relations'] += len(own_rel)
            summary['adapter_unique_relations'] += len(set(own_rel))
            summary['unique_relation_set_mismatch_images'] += int(set(canonical_rel) != set(own_rel))
            if canonical_rel != own_rel:
                summary['relation_multiset_mismatch_images'] += 1
                differences.append('relations')
            if differences and len(summary['examples']) < 5:
                summary['examples'].append({'image_id': identity, 'differences': differences})
        out['splits'][split] = summary
        del coco, annotations
    out['strict_release_parity'] = not out['predicate_vocabulary_mismatches'] and all(
        info['id_sets_equal'] and not info['object_vocabulary_mismatches']
        and info['object_count_mismatch_images'] == 0 and info['ordered_object_labels_mismatch_images'] == 0
        and info['dimension_mismatch_images'] == 0 and info['ordered_native_coco_clipped_boxes_mismatch_images'] == 0
        and info['relation_multiset_mismatch_images'] == 0 for info in out['splits'].values())
out['status'] = 'metadata_audit_complete'
out['published_protocol_claim'] = 'requires native evaluator and recipe parity in addition to these metadata checks'
text = json.dumps(out, indent=2)
(root / 'reports/VG_CANONICAL_PARITY_AUDIT.json').write_text(text)
print(text)
