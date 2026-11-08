import json
from pathlib import Path

shared = Path.home() / 'shared-assets/datasets/coco_karpathy'
p = shared / 'annotations/dataset_coco.json'

print("Inspecting dataset_coco.json...")
# Read first few entries without full memory overhead or load json
with open(p, 'r') as f:
    data = json.load(f)

print("Keys:", data.keys())
images = data['images']
print(f"Total images in dataset_coco: {len(images)}")

splits = {}
for im in images:
    s = im.get('split', 'unknown')
    splits[s] = splits.get(s, 0) + 1

print("Split distribution:")
for k, v in sorted(splits.items()):
    print(f"  {k}: {v} images")

sample = images[0]
print("\nSample image entry:")
print(json.dumps(sample, indent=2))

