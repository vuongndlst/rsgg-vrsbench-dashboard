from pathlib import Path
shared = Path.home() / 'shared-assets/datasets/coco_karpathy'

print("=== ANNOTATIONS DIR ===")
ann_dir = shared / 'annotations'
if ann_dir.exists():
    for f in ann_dir.iterdir():
        print(f"  {f.name} ({f.stat().st_size / 1e6:.1f} MB)" if f.is_file() else f"  [DIR] {f.name}")
else:
    print("  annotations dir not found")

print("=== IMAGES COUNT ===")
img_dir = shared / 'images'
if img_dir.exists():
    count = sum(1 for _ in img_dir.iterdir())
    print(f"  Total images in {img_dir}: {count}")

