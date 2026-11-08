import os, sys
from pathlib import Path

hub_dir = Path.home() / ".cache/huggingface/hub"
print("Hub dir:", hub_dir)
for p in (hub_dir / ".locks").rglob("*.lock"):
    print(p, p.stat().st_size, p.stat().st_mtime)

