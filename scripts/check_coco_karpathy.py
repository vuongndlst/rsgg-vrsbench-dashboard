import os, subprocess
from pathlib import Path

coco_dir = Path.home() / 'shared-assets/datasets/coco_karpathy'
print("Checking coco_dir:", coco_dir)
if coco_dir.exists():
    res = subprocess.run(['du', '-sh', str(coco_dir)], capture_output=True, text=True)
    print("Size of coco_dir:\n" + res.stdout)
    for p in coco_dir.glob('*'):
        if p.is_file():
            print(f"  File: {p.name} ({p.stat().st_size / (1024*1024):.1f} MB)")
        elif p.is_dir():
            res_sub = subprocess.run(['du', '-sh', str(p)], capture_output=True, text=True)
            print(f"  Dir: {p.name} -> {res_sub.stdout.strip()}")
else:
    print("coco_dir does not exist!")

# Also check other coco directories
for p in (Path.home() / 'shared-assets/datasets').glob('*'):
    print("Dataset dir:", p.name)

