import os, subprocess
from pathlib import Path

p = Path.home() / "etechs/msvd-tkg-captioning/models/qwen2.5-14b"
print(f"Path: {p}")
print(f"Exists: {p.exists()}")
if p.exists():
    for f in list(p.glob("*"))[:15]:
        print(f"  {f.name} ({f.stat().st_size / (1024*1024):.1f} MB)")
    total_size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / (1024*1024*1024)
    print(f"Total size: {total_size:.2f} GB")

