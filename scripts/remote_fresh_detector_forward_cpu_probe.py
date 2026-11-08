"""Low-resolution synthetic full detector/autograd smoke; no VG images or GPU."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import resource
import signal
import sys
import time

signal.alarm(120)
root = Path.home() / 'rsg-former'
sys.path.insert(0, str(root / 'src'))
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
started = time.monotonic()
from rsgformer.vg_fresh_detector import build_fresh_detector
model, criterion, post, config, provenance = build_fresh_detector(root)
from util.misc import nested_tensor_from_tensor_list
model.train()
torch.manual_seed(19283)
# Deliberately small invented tensor: functional compatibility only, never the
# native 800/1333 resolution recipe or GPU throughput/resource certification.
height, width = 128, 160
image = torch.randn(3, height, width)
samples = nested_tensor_from_tensor_list([image])
targets = [{'labels': torch.tensor([0, 74, 149]),
            'boxes': torch.tensor([[.2, .3, .15, .2], [.55, .5, .3, .25], [.8, .75, .12, .15]])}]
forward_started = time.monotonic()
outputs = model(samples)
forward_seconds = time.monotonic() - forward_started
assert outputs['pred_logits'].shape == (1, 200, 150)
assert outputs['pred_boxes'].shape == (1, 200, 4)
assert len(outputs['aux_outputs']) == 5
assert torch.isfinite(outputs['pred_logits']).all() and torch.isfinite(outputs['pred_boxes']).all()
losses = criterion(outputs, targets)
loss = sum(losses[key] * weight for key, weight in criterion.weight_dict.items() if key in losses)
assert torch.isfinite(loss)
backward_started = time.monotonic()
loss.backward()
backward_seconds = time.monotonic() - backward_started
groups = {'backbone': 0, 'transformer': 0, 'class_head': 0, 'box_head': 0}
nonzero_groups = {key: 0 for key in groups}
gradient_tensors = 0
for name, parameter in model.named_parameters():
    if parameter.grad is None:
        continue
    assert torch.isfinite(parameter.grad).all(), name
    gradient_tensors += 1
    # Native heads are registered under transformer.decoder; classify the
    # specific head name before its enclosing transformer prefix.
    group = next((key for key, token in [('class_head', 'class_embed'), ('box_head', 'bbox_embed'),
                                       ('backbone', 'backbone'), ('transformer', 'transformer')]
                  if token in name), None)
    if group:
        groups[group] += 1
        nonzero_groups[group] += int(bool(torch.count_nonzero(parameter.grad)))
assert all(count > 0 for count in nonzero_groups.values()), nonzero_groups
predictions = post['bbox'](outputs, torch.tensor([[height, width]]))
assert len(predictions[0]['scores']) == 100
assert torch.isfinite(predictions[0]['boxes']).all()
peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
assert peak_rss < 4 * 1024 ** 3, 'synthetic_CPU_memory_gate'
report = dict(provenance, status='PASS', utc=datetime.now(timezone.utc).isoformat(),
              scope='single invented low-resolution image, actual full model/criterion autograd',
              image_shape=list(image.shape), output_shapes={key: list(outputs[key].shape)
                                                          for key in ['pred_logits', 'pred_boxes']},
              synthetic_weighted_loss=float(loss.detach()), auxiliary_outputs=5,
              finite_gradient_tensors=gradient_tensors, gradient_tensors_by_group=groups,
              nonzero_gradient_tensors_by_group=nonzero_groups,
              forward_seconds=forward_seconds, backward_seconds=backward_seconds,
              elapsed_seconds=time.monotonic() - started, peak_rss_bytes=peak_rss,
              torch_cpu_threads=1, cpu_affinity_cores=1,
              cuda_used=False, actual_detector_forward_executed=True,
              actual_detector_backward_executed=True, optimizer_step_executed=False,
              dataset_training_started=False, vg_images_read=False, checkpoint_saved=False,
              benchmark_validation_scored=False, benchmark_test_scored=False,
              live_v2_modified=False,
              builder_sha256=hashlib.sha256((root / 'src/rsgformer/vg_fresh_detector.py').read_bytes()).hexdigest(),
              remaining=['Native-resolution GPU forward/backward/optimizer throughput and aggregate resource gate',
                         'Final external pixel overlap/exclusion release and separate training preregistration',
                         'Actual fresh detector dataset training and held-out evaluation'])
(root / 'reports/FRESH_DETECTOR_FORWARD_CPU_PROBE.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report))
