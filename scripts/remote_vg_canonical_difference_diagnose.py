"""Train-only annotation diagnostic of coordinate/dedup conventions; no scoring."""
from pathlib import Path
import os
import sys
import json
import zipfile
from collections import defaultdict, Counter
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
root = Path.home() / 'rsg-former'
sys.path[:0] = [str(root / 'src'), str(root / '.tools/deps')]
from rsgformer.data import VG150
ds = VG150(root / 'data/vg150', Path.home() / 'sbir/hienlth/vg', 'train', train_overlap=False)
with zipfile.ZipFile(root / 'data/vg_canonical_audit/reltr_vg_annotations.zip') as zipped:
    coco = json.loads(zipped.read('vg/train.json'))
    rel = json.loads(zipped.read('vg/rel.json'))['train']
anns = defaultdict(list)
for ann in coco['annotations']: anns[int(ann['image_id'])].append(ann)
out = {'mode': 'train_metadata_diagnostic_no_scoring', 'images': [], 'train_first1000': {
    'unique_triplet_set_mismatches': 0, 'pair_predicate_set_mismatches': 0,
    'inclusive_bbox_conversion_mismatch_images': 0, 'native_gt_relations': 0, 'canonical_relations': 0}}
for index, row in enumerate(ds.indices[:1000]):
    sample = ds._sample(row, clip=False)
    canonical = np.array([ann['bbox'] for ann in anns[sample.image_id]], dtype=np.float64)
    canonical[:, 2:] += canonical[:, :2] - 1
    canon_rel = [tuple(r) for r in rel[str(sample.image_id)]]
    own = [tuple(int(v) for v in r) for r in sample.relations]
    own_set = set(own); canonical_set = set(canon_rel)
    box_error = float(np.abs(canonical - sample.boxes).max())
    pairs_own = defaultdict(set); pairs_canon = defaultdict(set)
    for s, o, p in own: pairs_own[(s, o)].add(p)
    for s, o, p in canon_rel: pairs_canon[(s, o)].add(p)
    stats = out['train_first1000']
    stats['unique_triplet_set_mismatches'] += int(own_set != canonical_set)
    stats['pair_predicate_set_mismatches'] += int(dict(pairs_own) != dict(pairs_canon))
    stats['inclusive_bbox_conversion_mismatch_images'] += int(box_error > .001)
    stats['native_gt_relations'] += len(own); stats['canonical_relations'] += len(canon_rel)
    if index < 5:
        out['images'].append({'image_id': sample.image_id, 'first_canonical_bbox_xywh': anns[sample.image_id][0]['bbox'],
                             'first_h5_inclusive_xyxy': sample.boxes[0].tolist(), 'inclusive_box_max_error': box_error,
                             'own_relations': len(own), 'canonical_relations': len(canon_rel), 'own_unique': len(own_set),
                             'canonical_unique': len(canonical_set), 'canonical_only': sorted(canonical_set - own_set)[:8],
                             'own_only': sorted(own_set - canonical_set)[:8]})
text = json.dumps(out, indent=2)
(root / 'reports/VG_CANONICAL_DIFFERENCE_DIAGNOSIS.json').write_text(text)
print(text)
