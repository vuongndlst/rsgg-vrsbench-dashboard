import json
from pathlib import Path

ann_file = Path.home() / 'shared-assets/datasets/coco_karpathy/annotations/dataset_coco.json'
data = json.loads(ann_file.read_text())

for img in data['images']:
    if img['split'] == 'train':
        print("Sample train record:")
        print(f"  filename: {img.get('filename')}")
        print(f"  filepath: {img.get('filepath')}")
        print(f"  cocoid: {img.get('cocoid')}")
        print(f"  sentences: {len(img.get('sentences', []))}")
        break

