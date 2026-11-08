import subprocess
from pathlib import Path

p = Path.home() / 'shared-assets/datasets/coco_karpathy'
res = subprocess.run(['du', '-h', '--max-depth=2', str(p)], capture_output=True, text=True)
print("du output:\n" + res.stdout)

