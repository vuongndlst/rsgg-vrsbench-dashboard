import json
from pathlib import Path
from PIL import Image

SHARED = Path.home() / 'shared-assets/datasets/coco_karpathy'
ROOT = Path.home() / 'rsg-former'
img_dir = SHARED / 'images'

# Check train data
train_data_path = ROOT / 'artifacts/coco_val500_sgg_train_dataset.json'
raw_train = json.loads(train_data_path.read_text())
print(f"Total train images: {len(raw_train)}")

# Check test manifest
manifest = json.loads((SHARED / 'test_manifest.json').read_text())
test_images = manifest['images'][:100]
print(f"Test images in manifest: {len(test_images)}")

sgg_cache = json.loads((ROOT / 'artifacts/coco_5000_sgg_native_cache.json').read_text())
print(f"SGG cache entries: {len(sgg_cache)}")

# Check image existence for test
missing = sum(1 for e in test_images if not (img_dir / e['file_name']).exists())
print(f"Missing test images: {missing}")

# Check test refs
refs = json.loads((SHARED / 'test_references_coco.json').read_text())
print(f"Test refs annotations: {len(refs['annotations'])}")
print("PREFLIGHT SUCCESSFUL!")

