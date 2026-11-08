import os, glob
from pathlib import Path

print("=== SEARCHING FOR COCO FILES ON SERVER ===")
HOME = Path.home()
cands = []

# Search home, shared-assets, rsg-former
search_dirs = [HOME, HOME / 'shared-assets', HOME / 'rsg-former', Path('/tmp')]
for sdir in search_dirs:
    if not sdir.exists(): continue
    for p in sdir.glob('**/*coco*'):
        if any(skip in str(p) for skip in ['venv', '.git', 'site-packages', 'fasterrcnn_resnet50_fpn_v2_coco']):
            continue
        try:
            if p.is_file():
                cands.append((p, p.stat().st_size))
            elif p.is_dir() and 'coco' in p.name.lower():
                sz = sum(f.stat().st_size for f in p.rglob('*') if f.is_file())
                cands.append((p, sz))
        except Exception:
            pass

cands.sort(key=lambda x: x[1], reverse=True)
for p, sz in cands[:25]:
    print(f"{p}: {sz / (1024*1024):.1f} MB")

