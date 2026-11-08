"""Native VG postprocessing for the next canonical experiment, independent of GT."""
import numpy as np
import torch
import torchvision


@torch.no_grad()
def native_vg_postprocess(outputs, width, height):
    boxes = outputs['pred_boxes'][0]
    logits = outputs['pred_logits'][0].sigmoid()
    object_scores, classes = logits.softmax(-1).max(-1)
    relations = outputs['pred_rel'][0].sigmoid().clamp(0, 1)
    confidence = outputs['pred_confidence'][-1, 0].sigmoid().clamp(0, 1)
    n = len(boxes)
    assert logits.shape == (n, 150) and relations.shape == (n, n, 50)
    assert confidence.shape == (n, n) and width > 0 and height > 0
    for value in (boxes, logits, relations, confidence):
        if not torch.isfinite(value).all():
            raise ValueError('nonfinite_native_postprocessor_input')
    diagonal = torch.arange(n, device=boxes.device)
    relations[diagonal, diagonal] = 0
    confidence[diagonal, diagonal] = 0
    pair_object_scores = torch.outer(object_scores, object_scores) * confidence
    pair_object_scores[diagonal, diagonal] = 0
    xyxy = torch.cat((boxes[:, :2] - boxes[:, 2:] / 2,
                      boxes[:, :2] + boxes[:, 2:] / 2), dim=1)
    kept = torchvision.ops.batched_nms(xyxy, object_scores, classes, .5)
    overlaps = torchvision.ops.box_iou(xyxy, xyxy[kept])
    # Upstream stores positions within `kept`, solely as dedup equivalence IDs.
    # Evaluation retains the original query pairs/boxes, not merged coordinates.
    assignments = torch.zeros_like(classes)
    for category in torch.unique(classes[kept]):
        inputs = torch.where(classes == category)[0]
        survivors = torch.where(classes[kept] == category)[0]
        assignments[inputs] = survivors[overlaps[inputs][:, survivors].argmax(-1)]
    pairs = torch.cartesian_prod(diagonal, diagonal)
    mapped_pairs = assignments[pairs]
    predicate_labels = relations.max(-1)[1].reshape(-1)
    ranking = torch.mul(relations.max(-1)[0], pair_object_scores).reshape(-1)
    _, order = torch.sort(ranking, descending=True)
    tuples = torch.cat((mapped_pairs[order], predicate_labels[order, None]), -1)
    _, inverse, counts = torch.unique(tuples, dim=0, sorted=True, return_inverse=True, return_counts=True)
    _, grouped_indices = torch.sort(inverse, stable=True)
    offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)[:-1]))
    first = grouped_indices[offsets]
    mask = torch.zeros(len(order), dtype=torch.bool, device=boxes.device)
    mask[first] = True
    chosen = order[mask][:100]
    scale = xyxy.new_tensor([width, height, width, height])
    return {'pred_boxes': (xyxy * scale).cpu().numpy(),
            'pred_classes': classes.cpu().numpy(), 'obj_scores': object_scores.cpu().numpy(),
            'pred_rel_inds': pairs[chosen].cpu().numpy(),
            'rel_scores': relations.reshape(-1, 50)[chosen].cpu().numpy(),
            'ranking_scores': ranking[chosen].cpu().numpy(),
            'query_to_nms_equivalence_id': assignments.cpu().numpy(),
            'native_zero_score_rows_retained': int((ranking[chosen] == 0).sum())}
