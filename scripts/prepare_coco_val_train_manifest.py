import json
from pathlib import Path

ann_file = Path.home() / 'shared-assets/datasets/coco_karpathy/annotations/dataset_coco.json'
img_dir = Path.home() / 'shared-assets/datasets/coco_karpathy/images'

data = json.loads(ann_file.read_text())

train_pairs = []
available_val_imgs = []

for img in data['images']:
    if img['split'] == 'val':
        filename = img['filename']
        filepath = img_dir / filename
        if filepath.exists():
            available_val_imgs.append({
                'id': img['cocoid'],
                'file_name': filename,
                'captions': [s['raw'].strip() for s in img['sentences']]
            })

print(f"Total available validation images for training: {len(available_val_imgs)}")
print(f"Total training pairs (5 captions each): {len(available_val_imgs) * 5}")
sample = available_val_imgs[0]
print("Sample:", sample['file_name'], sample['id'])
for c in sample['captions'][:2]:
    print("  GT:", c)

# Save training manifest
out_path = Path.home() / 'rsg-former/data/coco_val500_train_manifest.json'
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(available_val_imgs, indent=2))
print(f"Saved manifest to {out_path}")

