import os
from pathlib import Path

ROOT = Path.home() / 'rsg-former'
art_dir = ROOT / 'artifacts'

# Old v1 epochs 1..4 can be safely removed since we have epoch 5 (final v1) and v2
to_remove = [
    art_dir / 'vrsbench_detector_epoch1.pth',
    art_dir / 'vrsbench_detector_epoch2.pth',
    art_dir / 'vrsbench_detector_epoch3.pth',
    art_dir / 'vrsbench_detector_epoch4.pth',
]

freed = 0
for f in to_remove:
    if f.exists():
        sz = f.stat().st_size
        f.unlink()
        freed += sz
        print(f"Removed {f.name} ({sz / (1024*1024):.1f} MB)")

print(f"\nTotal space freed: {freed / (1024*1024):.1f} MB ({freed / (1024*1024*1024):.2f} GB)")

import subprocess
res = subprocess.run(['df', '-h', '/'], capture_output=True, text=True)
print("Updated disk space:\n" + res.stdout)

