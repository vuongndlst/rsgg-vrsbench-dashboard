from pathlib import Path
zip_file = Path.home() / 'shared-assets/datasets/coco_karpathy/train2014.zip'
aria_file = Path.home() / 'shared-assets/datasets/coco_karpathy/train2014.zip.aria2'

if zip_file.exists():
    print(f"train2014.zip size: {zip_file.stat().st_size / 1e6:.1f} MB ({zip_file.stat().st_size / 1e9:.2f} GB)")
else:
    print("train2014.zip not found yet")

if aria_file.exists():
    print(f"train2014.zip.aria2 exists (download active)")

