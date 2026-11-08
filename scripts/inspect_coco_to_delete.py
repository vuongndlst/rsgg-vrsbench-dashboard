import os, subprocess
from pathlib import Path

img_dir = Path.home() / 'shared-assets/datasets/coco_karpathy/images'
print("Checking images in coco_karpathy/images:")
if img_dir.exists():
    items = list(img_dir.iterdir())
    print(f"Total items in images: {len(items)}")
    for it in items[:10]:
        print(f"  {it.name}: {it.stat().st_size / (1024*1024):.1f} MB (is_dir: {it.is_dir()})")

# Also check for any train2014.zip or coco zip archives in shared-assets or scratch
for z in (Path.home() / 'shared-assets/datasets').rglob('*.zip'):
    if 'coco' in str(z).lower():
        print(f"Found zip: {z} ({z.stat().st_size / (1024*1024):.1f} MB)")

