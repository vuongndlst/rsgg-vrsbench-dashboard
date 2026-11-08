import json
from pathlib import Path
from PIL import Image

SHARED = Path.home() / 'shared-assets/datasets/coco_karpathy'
img_dir = SHARED / 'images'

# Check train data
train_data_path = Path.home() / 'rsg-former/artifacts/coco_val500_sgg_train_dataset.json'
raw_train = json.loads(train_data_path.read_text())
print(f"Total train images: {len(raw_train)}")

train_missing = 0
for item in raw_train:
    p = img_dir / item['file_name']
    if not p.exists():
        train_missing += 1
print(f"Train images missing in {img_dir}: {train_missing}")

# Check test pilot 100
pilot_path = Path.home() / 'rsg-former/artifacts/coco_100_sgg_pilot_cache.json'
raw_test = json.loads(pilot_path.read_text())
print(f"Total test pilot images: {len(raw_test)}")

test_missing = 0
for item in raw_test:
    p = img_dir / item['file_name']
    if not p.exists():
        test_missing += 1
print(f"Test pilot images missing in {img_dir}: {test_missing}")

# Test opening one image
sample_img = img_dir / raw_train[0]['file_name']
im = Image.open(sample_img)
print(f"Sample image {sample_img.name} size: {im.size}, mode: {im.mode}")

