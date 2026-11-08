import json
from pathlib import Path
from collections import Counter

data_dir = Path.home() / 'rsg-former' / 'data' / 'ucm_captions'
data = json.loads((data_dir / 'dataset.json').read_text(encoding='utf-8'))
images = data.get('images', [])
splits = Counter(img.get('split') for img in images)
print("UCM total images:", len(images))
print("UCM splits:", splits)
print("Sample UCM 0:", images[0])
