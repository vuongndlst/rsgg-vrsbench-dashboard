"""CPU-only train metadata/image contract probe; no model, GPU, or test scoring."""
from pathlib import Path
from datetime import datetime, timezone
import ast
import hashlib
import inspect
import json
import os
import sys
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
root = Path.home() / 'rsg-former'
sys.path[:0] = [str(root / 'src'), str(root / '.tools/deps')]
import numpy as np
import torch
torch.set_num_threads(1)
from PIL import Image
from torchvision.datasets import CocoDetection
from rsgformer.vg_canonical_data import VGCanonical

source_path = root / 'vendor/salience_sgg/datasets/coco.py'
source = source_path.read_text()
tree = ast.parse(source)
cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'ConvertCocoPolysToMask')
namespace = {'torch': torch}
exec(compile(ast.Module(body=[cls], type_ignores=[]), str(source_path), 'exec'), namespace)
prepare = namespace['ConvertCocoPolysToMask'](False)
constructor = inspect.getsource(CocoDetection.__init__)
assert 'sorted' in constructor and 'self.coco.imgs' in constructor, 'Iteration-order contract changed'
ds = VGCanonical(root / 'data/vg_canonical_audit/reltr_vg_annotations.zip', Path.home() / 'sbir/hienlth/vg', 'train')
count = 0
for index in range(len(ds)):
    sample = ds[index]
    meta = ds.image_info[sample.image_id]
    # Actual target conversion from pinned upstream, using only image dimensions.
    image = Image.new('RGB', (sample.width, sample.height))
    _, target = prepare(image, {'image_id': sample.image_id, 'annotations': ds.annotations[sample.image_id]})
    assert np.array_equal(target['boxes'].numpy(), sample.boxes)
    assert np.array_equal(target['labels'].numpy(), sample.labels)
    if index < 8 and len(sample.labels) <= 200:
        native = ds.native_target(index)
        assert torch.isfinite(native['boxes']).all() and native['rel'].shape == (200, 200, 50)
        assert int(native['rel'].sum()) == len(sample.relations)
    count += 1
    image.close()
decoded = []
for index in [0, 1]:
    image = ds.image(index)
    decoded.append({'image_id': ds[index].image_id, 'dimensions': list(image.size)})
    image.close()
out = {'status': 'PASS', 'utc': datetime.now(timezone.utc).isoformat(),
       'scope': 'all canonical TRAIN COCO target contracts; two TRAIN ZIP JPEG reads',
       'images_checked': count, 'train_iteration': 'torchvision sorted image ID',
       'train_sorted_ids_sha256': ds.split_hash(), 'decoded_train_images': decoded,
       'adapter_sha256': hashlib.sha256((root / 'src/rsgformer/vg_canonical_data.py').read_bytes()).hexdigest(),
       'upstream_coco_source_sha256': hashlib.sha256(source_path.read_bytes()).hexdigest(),
       'annotation_sha256': ds.annotation_sha256, 'relation_sha256': ds.relation_sha256,
       'model_initialized': False, 'cuda_used': False, 'image_archives_extracted': False,
       'validation_performance_scored': False, 'test_performance_scored': False,
       'live_v2_modified': False, 'clean_release_certified': False,
       'remaining': ['native train augmentation/criterion/matcher contracts', 'canonical SG evaluator/postprocessing parity',
                     'external held-out identity/pixel audit', 'audited fresh initialization', 'resource smoke before fresh detector training']}
ds.close()
text = json.dumps(out, indent=2)
(root / 'reports/VG_CANONICAL_ADAPTER_PROBE.json').write_text(text)
print(text)
