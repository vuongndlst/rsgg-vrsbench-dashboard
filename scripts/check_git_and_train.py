import os, sys, glob, urllib.request
from pathlib import Path

print("=== CHECKING LOCAL MODELS ===")
home = Path.home()
patterns = [
    str(home / 'etechs/**/models/*'),
    str(home / 'models/*'),
    str(home / '.cache/huggingface/hub/*'),
    '/home/stackops/etechs/msvd-tkg-captioning/models/*',
    str(home / 'shared-assets/**/*')
]
for p in patterns:
    matches = glob.glob(p)
    if matches:
        print(f"Pattern {p}: {len(matches)} matches")
        for m in matches[:5]:
            print(f"  {m}")

print("\n=== CHECKING COCO TRAINING ASSETS ===")
coco_dir = home / 'shared-assets/datasets/coco_karpathy'
if coco_dir.exists():
    for f in coco_dir.iterdir():
        print(f"  {f.name} ({f.stat().st_size / 1e6:.1f} MB)" if f.is_file() else f"  [DIR] {f.name}")

print("\n=== CHECKING INTERNET / HUGGINGFACE ACCESS ===")
try:
    with urllib.request.urlopen("https://huggingface.co", timeout=5) as conn:
        print(f"HuggingFace connectivity: OK (status {conn.status})")
except Exception as e:
    print(f"HuggingFace connectivity: FAILED ({e})")

