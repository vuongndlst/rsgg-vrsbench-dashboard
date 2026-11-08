"""Bounded canonical TRAIN augmentation/criterion integration; no model inference."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import pickle
import random
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
from rsgformer.vg_canonical_data import VGCanonical
from rsgformer.vg_canonical_training import VGCanonicalTraining

manifest_path = root / 'artifacts/caption_readiness/canonical_known_identity_clean_splits.json'
manifest = json.loads(manifest_path.read_text())
assert manifest['status'].startswith('PROVISIONAL_')
canonical = VGCanonical(root / 'data/vg_canonical_audit/reltr_vg_annotations.zip',
                        Path.home() / 'sbir/hienlth/vg', 'train',
                        exclude_image_ids=manifest['excluded_known_image_ids'])
assert canonical.ids == manifest['split_image_ids']['train']
assert canonical.split_hash() == manifest['provisional_sorted_split_sha256']['train']
_, criterion, _, _, provenance = build_fresh_detector(root)
records = []
for task in ('detection', 'sgg'):
    dataset = VGCanonicalTraining(canonical, root, task=task)
    for index in range(4):
        sample = canonical[index]
        for seed in range(4):
            random.seed(seed)
            torch.manual_seed(seed)
            image, target = dataset[index]
            count = len(target['labels'])
            assert torch.isfinite(image).all() and image.shape[0] == 3
            assert torch.isfinite(target['boxes']).all() and target['boxes'].shape == (count, 4)
            assert torch.all((target['labels'] >= 0) & (target['labels'] < 150))
            assert torch.all((target['boxes'] >= 0) & (target['boxes'] <= 1))
            assert len(target['area']) == count and len(target['iscrowd']) == count
            if task == 'sgg':
                assert count == len(sample.labels)  # pinned SGG recipe has no crop
                assert int(target['rel'].sum()) == len(sample.relations)
                assert target['rel'].shape == (200, 200, 50)
            logits = torch.randn(1, 200, 150, requires_grad=True)
            boxes = torch.rand(1, 200, 4, requires_grad=True)
            losses = criterion({'pred_logits': logits, 'pred_boxes': boxes}, [target])
            weighted = sum(value * criterion.weight_dict[key]
                           for key, value in losses.items() if key in criterion.weight_dict)
            assert torch.isfinite(weighted)
            weighted.backward()
            assert torch.isfinite(logits.grad).all() and torch.isfinite(boxes.grad).all()
            records.append({'task': task, 'image_id': sample.image_id, 'seed': seed,
                            'output_height_width': list(image.shape[-2:]), 'objects': count,
                            'source_objects': len(sample.labels)})
    # Spawn-worker pickle resets transient AST recipe classes and ZIP handles.
    restored = pickle.loads(pickle.dumps(dataset))
    assert restored._runtime is None and restored.canonical._zips is None
    random.seed(20)
    torch.manual_seed(20)
    image, target = restored[0]
    assert torch.isfinite(image).all()
    restored.canonical.close()

canonical.close()
names = ['src/rsgformer/vg_canonical_training.py', 'src/rsgformer/vg_canonical_data.py',
         'vendor/salience_sgg/datasets/coco.py', 'vendor/salience_sgg/datasets/visual_genome.py',
         'vendor/salience_sgg/datasets/transforms.py']
out = {'status': 'PASS', 'utc': datetime.now(timezone.utc).isoformat(),
       'scope': '4 known-identity-excluded TRAIN images, 4 RNG seeds, native detection and SGG recipes',
       'elapsed_seconds': time.monotonic() - started, 'cases': records,
       'provisional_train_images': len(canonical), 'provisional_train_ids_sha256': canonical.split_hash(),
       'provisional_manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
       'source_hashes': {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names},
       'detection_has_native_random_crop': True, 'sgg_has_native_random_crop': False,
       'native_criterion_finite_loss_and_gradients_on_augmented_targets': 'PASS',
       'spawn_worker_pickle_and_reopen': 'PASS', 'cuda_used': False, 'detector_forward_executed': False,
       'image_archives_extracted': False, 'test_scored': False, 'training_started': False,
       'clean_release_certified': False, 'vg_checkpoint_loaded': provenance['vg_checkpoint_loaded']}
text = json.dumps(out, indent=2)
(root / 'reports/CANONICAL_TRAINING_CPU_PROBE.json').write_text(text)
print(text)
