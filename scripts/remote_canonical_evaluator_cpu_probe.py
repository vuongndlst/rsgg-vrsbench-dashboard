"""Synthetic evaluator contracts only: no benchmark images/model/test metrics."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import sys
import signal
root = Path.home() / 'rsg-former'
sys.path.insert(0, str(root / 'src'))
signal.alarm(60)
import numpy as np
from rsgformer.data import ImageGraphSample
from rsgformer.vg_canonical_evaluation import CanonicalSGDetEvaluator, inclusive_iou_float64

# Scalar arithmetic copied from the pinned Cython loop, tested against vectorized
# float64 implementation; this is not a compiled-extension execution claim.
def scalar_iou(boxes, queries):
    out = np.zeros((len(boxes), len(queries)), dtype=float)
    for k, query in enumerate(queries):
        area = (query[2] - query[0] + 1) * (query[3] - query[1] + 1)
        for n, box in enumerate(boxes):
            iw = min(box[2], query[2]) - max(box[0], query[0]) + 1
            ih = min(box[3], query[3]) - max(box[1], query[1]) + 1
            if iw > 0 and ih > 0:
                union = (box[2] - box[0] + 1) * (box[3] - box[1] + 1) + area - iw * ih
                out[n, k] = iw * ih / union
    return out
rng = np.random.default_rng(8261)
points = rng.uniform(0, 100, (128, 2))
boxes = np.concatenate((points, points + rng.uniform(0, 80, (128, 2))), 1)
boxes = np.vstack((boxes, [0, 0, 1, 1], [2/3, 0, 5/3, 1], [5, 5, 5, 5]))
actual = inclusive_iou_float64(boxes, boxes[::-1])
formula_error = float(np.max(np.abs(actual - scalar_iou(boxes, boxes[::-1]))))
assert formula_error < 1e-12
s = ImageGraphSample(8261, 100, 100, np.array([[0,0,20,20],[30,30,50,50]], dtype=float),
                     np.array([1,150]), np.array([[0,1,1],[1,0,50]]), np.array([0,1]))
scores = np.zeros((2,50)); scores[0,0] = .9; scores[1,49] = .8
p = {'pred_boxes': s.boxes.copy(), 'pred_classes': s.labels-1, 'obj_scores': np.ones(2),
     'pred_rel_inds': np.array([[0,1],[1,0]]), 'rel_scores': scores, 'ranking_scores': np.array([.9,.8])}
results = {}
for case in ('perfect', 'wrong_object_class', 'wrong_predicate', 'wrong_box', 'empty_predictions'):
    pred = {key: value.copy() for key,value in p.items()}
    if case == 'wrong_object_class': pred['pred_classes'][:] = 77
    if case == 'wrong_predicate': pred['rel_scores'] = np.roll(pred['rel_scores'], 1, axis=1)
    if case == 'wrong_box': pred['pred_boxes'] += 70
    if case == 'empty_predictions':
        for key in ('pred_rel_inds','rel_scores','ranking_scores'): pred[key] = pred[key][:0]
    evaluator = CanonicalSGDetEvaluator(root); evaluator.update(s,pred)
    result = evaluator.result()
    assert result['R@50'] == (1.0 if case == 'perfect' else 0.0)
    assert result['mR@50'] == (.04 if case == 'perfect' else 0.0)
    results[case] = {'R@50': result['R@50'], 'mR@50': result['mR@50']}
evaluator = CanonicalSGDetEvaluator(root)
evaluator.update(s,p)
try: evaluator.update(s,p)
except ValueError as error: assert str(error) == 'duplicate_image_evaluation'
else: raise AssertionError('duplicate image accepted')
bad = dict(p, ranking_scores=np.array([.1,.9]))
try: CanonicalSGDetEvaluator(root).update(s,bad)
except ValueError as error: assert str(error) == 'predictions_not_in_registered_rank_order'
else: raise AssertionError('unranked predictions accepted')
names = ['src/rsgformer/vg_canonical_evaluation.py', 'vendor/salience_sgg/evaluation/vg_sgg_evaluator.py',
         'vendor/salience_sgg/lib/fpn/box_intersections_cpu/bbox.pyx']
out = {'status':'PASS', 'utc':datetime.now(timezone.utc).isoformat(),
       'scope':'synthetic matching, class/predicate endpoints, empty misses and float64 IoU arithmetic',
       'scalar_pinned_cython_formula_max_abs_error':formula_error, 'synthetic_cases':results,
       'source_hashes':{name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names},
       'compiled_bbox_extension_executed':False, 'cuda_used':False, 'model_initialized':False,
       'benchmark_validation_scored':False, 'benchmark_test_scored':False, 'live_v2_modified':False,
       'remaining':['Native postprocessor/ranking/NMS parity', 'Canonical release and clean-training preregistration',
                    'Real model validation after training; synthetic R/mR are not benchmark evidence']}
text=json.dumps(out,indent=2)
(root/'reports/CANONICAL_EVALUATOR_CPU_PROBE.json').write_text(text)
print(text)
