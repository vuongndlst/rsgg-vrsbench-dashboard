"""Inspect metadata headers and initialization cache, never caption values or weights."""
from pathlib import Path
import csv
import json
root = Path.home() / 'shared-assets/datasets/openimages_test/annotations'
out = {}
for name in ['test-images-with-rotation.csv', 'oidv6-test-annotations-vrd.csv']:
    with (root / name).open(encoding='utf8', newline='') as stream:
        reader = csv.DictReader(stream)
        out[name] = {'columns': reader.fieldnames}
with (root / 'open_images_test_captions.jsonl').open(encoding='utf8') as stream:
    item = json.loads(next(stream))
    out['caption_columns'] = sorted(item)
cached = Path.home() / '.cache/torch/hub/checkpoints'
out['resnet50_cache'] = [{'name': p.name, 'bytes': p.stat().st_size} for p in cached.glob('resnet50*.pth') if p.is_file()]
print(json.dumps(out, indent=2))
