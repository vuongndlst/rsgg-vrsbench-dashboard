import subprocess, os
from pathlib import Path

res = subprocess.run(['df', '-h'], capture_output=True, text=True)
print("DISK USAGE:\n" + res.stdout)

print("\nARTIFACTS CHECKPOINTS USAGE:")
ROOT = Path.home() / 'rsg-former'
ckpts = sorted((ROOT / 'artifacts').glob('*.pth'))
total_bytes = 0
for c in ckpts:
    sz = c.stat().st_size
    total_bytes += sz
    print(f"  {c.name}: {sz / (1024*1024):.1f} MB")
print(f"Total .pth files size: {total_bytes / (1024*1024*1024):.2f} GB")

