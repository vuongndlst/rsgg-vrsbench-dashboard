import os, sys, subprocess, glob
from pathlib import Path

print("=== DISK SPACE ===")
print(subprocess.getoutput("df -h /home/stackops"))

print("\n=== RUNNING PROCESSES (aria2 / python) ===")
ps = subprocess.getoutput("ps aux | grep -E 'aria2|python|unzip' | grep -v grep")
print(ps if ps else "No matching processes.")

print("\n=== DATASET DIRECTORY CONTENTS ===")
base = Path.home() / 'shared-assets/datasets/coco_karpathy'
if base.exists():
    for p in base.iterdir():
        if p.is_file():
            print(f"File: {p.name} ({p.stat().st_size / (1024**2):.2f} MB)")
        elif p.is_dir():
            if p.name == 'images':
                sub = list(p.iterdir())
                print(f"Dir: images/ containing {[s.name for s in sub]}")
                for s in sub:
                    if s.is_dir():
                        cnt = len(list(s.glob('*.jpg')))
                        print(f"  - {s.name}: {cnt} images")
            else:
                print(f"Dir: {p.name}/")
else:
    print(f"{base} does not exist!")

