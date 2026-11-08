"""Separate cached-head batching experiment; unchanged per-image criterion."""
import torch

KEYS = ('nodes', 'boxes', 'logits', 'visual')

def pack(items, device='cuda', pinned=False):
    result = {}
    for key in KEYS:
        value = torch.stack([item[key].float() for item in items])
        if pinned:
            value = value.pin_memory()
        result[key] = value.to(device, non_blocking=pinned)
    return result

def target(sample, device='cuda'):
    # Keep GPU normalization and indexing in the same order as locked v2.
    box = torch.tensor(sample.boxes, device=device) / torch.tensor(
        [sample.width, sample.height, sample.width, sample.height], device=device)
    rel = torch.zeros(200, 200, 50, device=device)
    indices = torch.tensor(sample.relations, device=device)
    rel[indices[:, 0], indices[:, 1], indices[:, 2]-1] = 1
    return {'labels': torch.tensor(sample.labels-1, device=device),
            'boxes': torch.cat([(box[:, :2]+box[:, 2:])/2, box[:, 2:]-box[:, :2]], -1),
            'rel': rel}

def singleton(prediction, index):
    # Confidence is [salience_layer, batch, query, query], unlike other outputs.
    return {key: value[:, index:index+1] if key == 'pred_confidence'
            else value[index:index+1] for key, value in prediction.items()}

def per_image_losses(criterion, prediction, targets):
    values = []
    for index, gt in enumerate(targets):
        losses = criterion(singleton(prediction, index), [gt])
        values.append(sum(losses[key]*criterion.weight_dict[key]
                          for key in losses if key in criterion.weight_dict))
    return torch.stack(values)

def finite_loss(values):
    # Retain immediate non-finite checks; only logging synchronization is grouped.
    if not bool(torch.isfinite(values).all()):
        raise RuntimeError('non_finite_loss')

