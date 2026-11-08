"""Bounded fresh construction/criterion audit, no CUDA or detector forward."""
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

root = Path.home() / 'rsg-former'
sys.path.insert(0, str(root / 'src'))
import torch
torch.set_num_threads(1)
signal.alarm(180)
started = time.monotonic()
from rsgformer.vg_fresh_detector import build_fresh_detector
model, criterion, post, config, provenance = build_fresh_detector(root)

# Exercise actual native Hungarian matching, endpoint class IDs, empty targets,
# auxiliary decoder losses, finite autograd and native bbox postprocessing.
torch.manual_seed(7813)
logits = torch.full((2, 200, 150), -4.0)
logits[0, 7, 0] = 8
logits[0, 23, 149] = 8
logits.requires_grad_()
boxes = torch.full((2, 200, 4), .2)
boxes[0, 7] = torch.tensor([.15, .25, .1, .2])
boxes[0, 23] = torch.tensor([.8, .8, .2, .2])
boxes.requires_grad_()
targets = [
    {'labels': torch.tensor([149, 0]), 'boxes': boxes[0, [23, 7]].detach().clone()},
    {'labels': torch.empty(0, dtype=torch.int64), 'boxes': torch.empty((0, 4))},
]
outputs = {'pred_logits': logits, 'pred_boxes': boxes}
outputs['aux_outputs'] = [{'pred_logits': logits, 'pred_boxes': boxes} for _ in range(5)]
indices = criterion.matcher(outputs, targets)
assert indices[0][0].tolist() == [7, 23] and indices[0][1].tolist() == [1, 0]
assert len(indices[1][0]) == 0
losses = criterion(outputs, targets)
weighted = sum(losses[key] * weight for key, weight in criterion.weight_dict.items() if key in losses)
assert torch.isfinite(weighted)
weighted.backward()
assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0
assert torch.isfinite(boxes.grad).all()
assert all(f'loss_ce_{layer}' in losses for layer in range(5))
predictions = post['bbox'](outputs, torch.tensor([[600, 800], [600, 800]]))
assert len(predictions) == 2 and all(len(p['scores']) == 100 for p in predictions)
assert set(predictions[0]['labels'][:2].tolist()) == {0, 149}

source_names = ['src/rsgformer/vg_fresh_detector.py',
                'vendor/salience_sgg/models/dab_deformable_detr/deformable_detr.py',
                'vendor/salience_sgg/models/dab_deformable_detr/matcher.py',
                'vendor/salience_sgg/models/dab_deformable_detr/backbone.py',
                'vendor/salience_sgg/models/dab_deformable_detr/deformable_transformer.py',
                'vendor/salience_sgg/models/ops/functions/ms_deform_attn_func.py']
out = dict(provenance, status='PASS', utc=datetime.now(timezone.utc).isoformat(),
           elapsed_seconds=time.monotonic() - started,
           cuda_used=False, detector_forward_executed=False, training_started=False,
           test_scored=False, checkpoint_saved=False,
           criterion_synthetic_weighted_loss=float(weighted.detach()),
           hungarian_class_endpoints_swapped_assignment_empty_target='PASS',
           auxiliary_losses_and_finite_logit_gradients='PASS', native_bbox_postprocess='PASS',
           source_hashes={name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in source_names},
           remaining=['Actual canonical train transforms/targets criterion integration',
                      'Full fresh detector forward/backward GPU memory and throughput gate',
                      'Final pixel overlap/exclusion release and separate training preregistration',
                      'Fresh clean-train relation prior and SGG training/evaluation'])
text = json.dumps(out, indent=2)
(root / 'reports/FRESH_DETECTOR_CPU_PROBE.json').write_text(text)
print(text)
