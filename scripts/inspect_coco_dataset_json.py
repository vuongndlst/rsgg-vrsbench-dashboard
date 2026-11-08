import json, os
from pathlib import Path

ann_file = Path.home() / 'shared-assets/datasets/coco_karpathy/annotations/dataset_coco.json'
img_dir = Path.home() / 'shared-assets/datasets/coco_karpathy/images'

print(f"Reading {ann_file}...")
data = json.loads(ann_file.read_text())

splits = {}
available_images_per_split = {}
for img in data['images']:
    s = img['split']
    splits[s] = splits.get(s, 0) + 1
    
    # Check if image file exists on disk
    filename = img['filename']
    filepath = img_dir / filename
    if filepath.exists():
        available_images_per_split[s] = available_images_per_split.get(s, 0) + 1

print("Total images in dataset_coco.json by split:")
for s, cnt in splits.items():
    avail = available_images_per_split.get(s, 0)
    print(f"  Split '{s}': {cnt} images (Available on disk: {avail})")

# Sample annotations structure
sample = data['images'][0]
print("\nSample image record:")
print(f"  filename: {sample.get('filename')}")
print(f"  split: {sample.get('split')}")
print(f"  sentences: {len(sample.get('sentences', []))}")
for sent in sample.get('sentences', [])[:2]:
    print(f"    - {sent.get('raw')}")

