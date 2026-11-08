"""Pinned canonical SGDet GC evaluator bridge for a future experiment.

Requires predictions already ranked by the registered native postprocessor.
Keeps float64 inclusive IoU from the pinned Cython source. No live v2 edits.
"""
from pathlib import Path
import importlib.util
import sys
import types
import numpy as np


def inclusive_iou_float64(boxes, queries):
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    queries = np.asarray(queries, dtype=np.float64).reshape(-1, 4)
    if not (np.isfinite(boxes).all() and np.isfinite(queries).all()):
        raise ValueError('nonfinite_evaluator_box')
    if not ((boxes[:, 2:] >= boxes[:, :2]).all() and (queries[:, 2:] >= queries[:, :2]).all()):
        raise ValueError('invalid_evaluator_box')
    width_height = np.maximum(0, np.minimum(boxes[:, None, 2:], queries[None, :, 2:])
                              - np.maximum(boxes[:, None, :2], queries[None, :, :2]) + 1)
    intersection = width_height.prod(-1)
    areas = (boxes[:, 2:] - boxes[:, :2] + 1).prod(-1)
    query_areas = (queries[:, 2:] - queries[:, :2] + 1).prod(-1)
    return intersection / (areas[:, None] + query_areas[None, :] - intersection)


class CanonicalSGDetEvaluator:
    def __init__(self, root):
        vendor = Path(root) / 'vendor/salience_sgg'
        sys.path.insert(0, str(vendor))
        legacy = types.ModuleType('lib.fpn.box_intersections_cpu.bbox')
        legacy.bbox_overlaps = inclusive_iou_float64
        sys.modules['lib.fpn.box_intersections_cpu.bbox'] = legacy
        path = vendor / 'evaluation/vg_sgg_evaluator.py'
        specification = importlib.util.spec_from_file_location('canonical_native_vg_evaluator', path)
        reference = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(reference)
        self.native = reference.BasicSceneGraphEvaluator('sgdet', multiple_preds=False)
        self.predicates = [reference.BasicSceneGraphEvaluator('sgdet', multiple_preds=False) for _ in range(50)]
        self.records = []
        self.empty_gt_ids = []
        self.observed_ids = set()

    def update(self, sample, prediction):
        if sample.image_id in self.observed_ids:
            raise ValueError('duplicate_image_evaluation')
        self.observed_ids.add(sample.image_id)
        relations = np.asarray(sample.relations, dtype=np.int64).reshape(-1, 3).copy()
        if len(set(map(tuple, relations))) != len(relations):
            raise ValueError('canonical_gt_contains_duplicate_relations')
        if not len(relations):
            self.empty_gt_ids.append(sample.image_id)
            return
        gt = {'gt_relations': relations, 'gt_boxes': sample.boxes,
              'gt_classes': np.asarray(sample.labels, dtype=np.int64) - 1}
        gt['gt_relations'][:, 2] -= 1
        assert np.all((gt['gt_classes'] >= 0) & (gt['gt_classes'] < 150))
        assert np.all((relations[:, 2] >= 0) & (relations[:, 2] < 50))
        pairs = np.asarray(prediction['pred_rel_inds'], dtype=np.int64).reshape(-1, 2)
        scores = np.asarray(prediction['rel_scores']).reshape(-1, 50)
        ranking = np.asarray(prediction['ranking_scores']).reshape(-1)
        assert len(pairs) == len(scores) == len(ranking)
        assert np.isfinite(scores).all() and np.isfinite(ranking).all()
        if np.any(ranking[1:] > ranking[:-1]):
            raise ValueError('predictions_not_in_registered_rank_order')
        boxes = np.asarray(prediction['pred_boxes']).reshape(-1, 4)
        classes = np.asarray(prediction['pred_classes'], dtype=np.int64).reshape(-1)
        obj_scores = np.asarray(prediction['obj_scores']).reshape(-1)
        assert len(boxes) == len(classes) == len(obj_scores)
        assert np.all((classes >= 0) & (classes < 150))
        assert np.isfinite(boxes).all() and np.isfinite(obj_scores).all()
        assert np.all((pairs >= 0) & (pairs < len(boxes)))
        predicted = {'pred_rel_inds': pairs[:100], 'rel_scores': scores[:100],
                     'pred_boxes': boxes, 'pred_classes': classes, 'obj_scores': obj_scores}
        present = np.unique(relations[:, 2])
        targets = [(self.native, gt)]
        for predicate in present:
            filtered = dict(gt, gt_relations=relations[relations[:, 2] == predicate])
            targets.append((self.predicates[predicate], filtered))
        for evaluator, target in targets:
            if len(pairs):
                evaluator.evaluate_scene_graph_entry(target, predicted, iou_thresh=.5)
            else:
                # Upstream's zero-prediction helper returns a short tuple. Count a
                # miss explicitly rather than dropping images or modifying source.
                for values in evaluator.result_dict['sgdet_recall'].values():
                    values.append(0.0)
        record = {'image_id': sample.image_id, 'recalls': {}, 'predicates': {}}
        for k in (20, 50, 100):
            record['recalls'][str(k)] = self.native.result_dict['sgdet_recall'][k][-1]
            for predicate in present:
                record['predicates'].setdefault(str(predicate + 1), {})[str(k)] = \
                    self.predicates[predicate].result_dict['sgdet_recall'][k][-1]
        self.records.append(record)

    def result(self):
        result = {'mode': 'SGDet', 'constraint': 'GC', 'images': len(self.records),
                  'observed_images': len(self.observed_ids), 'empty_gt_images_excluded': len(self.empty_gt_ids),
                  'matching': 'pinned native class/predicate AND inclusive float64 IoU>=0.5',
                  'absent_predicate_policy': 'contributes zero in denominator50'}
        for k in (20, 50, 100):
            values = self.native.result_dict['sgdet_recall'][k]
            result[f'R@{k}'] = float(np.mean(values)) if values else None
            result[f'mR@{k}'] = sum(float(np.mean(e.result_dict['sgdet_recall'][k]))
                                    if e.result_dict['sgdet_recall'][k] else 0.0
                                    for e in self.predicates) / 50 if values else None
        return result
