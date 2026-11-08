import os
from pathlib import Path
from PIL import Image

p = Path.home() / 'rsg-former' / 'data' / 'sydney_captions' / 'images'
files = list(p.glob("*"))
print(f"Found {len(files)} files in {p}")
if files:
    img = Image.open(files[0])
    print(f"Sample image {files[0].name}: format={img.format}, size={img.size}, mode={img.mode}")

