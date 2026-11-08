import json
from pathlib import Path

data_dir = Path.home() / 'rsg-former' / 'data' / 'sydney_captions'
data = json.loads((data_dir / 'dataset.json').read_text(encoding='utf-8'))
print("Keys:", data.keys())
images = data.get('images', [])
print(f"Total images: {len(images)}")
print("Sample image 0:", images[0])
print("Sample image 1:", images[1])

