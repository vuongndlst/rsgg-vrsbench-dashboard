import json
from pathlib import Path
ROOT = Path.expanduser(Path('~/rsg-former'))
ds = json.loads((ROOT / 'data/ucm_captions/dataset.json').read_text(encoding='utf-8'))
print("Root keys:", ds.keys())
if 'images' in ds:
    print("Sample img 0:", {k: v for k, v in ds['images'][0].items() if k != 'sentences'})
    print("Sample img 100:", {k: v for k, v in ds['images'][100].items() if k != 'sentences'})
