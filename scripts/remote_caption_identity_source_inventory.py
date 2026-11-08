"""Read-only metadata identity gaps and bounded existing COCO source inventory."""
from pathlib import Path
from collections import deque, Counter
from datetime import datetime, timezone
import hashlib
import json
import os
import re
import signal

signal.alarm(60)
os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
home = Path.home()
root = home / 'rsg-former'
metadata_path = root / 'data/vg150/image_data.json'
raw_metadata = metadata_path.read_bytes()
metadata = json.loads(raw_metadata)
by_id = {int(item['image_id']): item for item in metadata}
assert len(by_id) == len(metadata)
manifest_path = root / 'artifacts/caption_readiness/canonical_known_identity_clean_splits.json'
manifest_bytes = manifest_path.read_bytes()
manifest = json.loads(manifest_bytes)

def category(item):
    coco = bool(item.get('coco_id'))
    flickr = bool(item.get('flickr_id'))
    return ('both' if coco and flickr else 'coco_only' if coco else 'flickr_only' if flickr else 'neither')

coverage = {'all_metadata': dict(Counter(category(item) for item in metadata))}
missing = {}
for split, identities in manifest['split_image_ids'].items():
    coverage[split] = dict(Counter(category(by_id[int(identity)]) for identity in identities))
    missing[split] = [int(identity) for identity in identities if not by_id[int(identity)].get('flickr_id')]

roots = [home / name for name in ['shared-assets/datasets', 'image-captioning-kg', 'fair-project',
                                  'coco_kb_v3', 'v2_kb_20260909', 'v2_kb', 'skgcap_vlm', 'etechs']]
pending = deque((folder, 0) for folder in roots)
visited = set()
candidates = {}
pattern = re.compile(r'^(?:instances|captions|image_info)_(?:train|val|test|test-dev)(?:2014|2017)\.json$')
while pending and len(visited) < 600:
    folder, depth = pending.popleft()
    try:
        resolved = folder.resolve()
        if str(resolved) in visited or not resolved.is_dir():
            continue
        visited.add(str(resolved))
        entries = list(folder.iterdir())[:2000]
    except OSError:
        continue
    for entry in entries:
        if entry.is_file() and pattern.fullmatch(entry.name):
            canonical = entry.resolve()
            candidates[str(canonical)] = {'path_alias': str(canonical).replace(str(home), '${REMOTE_HOME}'),
                                           'bytes': canonical.stat().st_size, 'filename': entry.name}
        elif depth < 6 and entry.is_dir():
            if entry.name in ['.git', '.ssh', '.aws', '.venv', 'models', 'weights', 'images',
                              'train2014', 'val2014', 'train2017', 'val2017', 'test2017',
                              'node_modules', 'site-packages', 'blobs', 'outputs', 'logs']:
                continue
            if depth == 0 or re.search(r'coco|data|annotation|project|assets|incoming|runtime|kb', entry.name, re.I):
                pending.append((entry, depth + 1))

target = root / 'artifacts/caption_readiness/vg_missing_flickr_identity_ids.json'
target.write_text(json.dumps({'scope': 'metadata gaps only; no new exclusions or clean certification',
                              'metadata_sha256': hashlib.sha256(raw_metadata).hexdigest(),
                              'provisional_manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
                              'missing_flickr_ids_by_split': missing}, separators=(',', ':')))
report = {'status': 'METADATA_SOURCE_INVENTORY_COMPLETE', 'utc': datetime.now(timezone.utc).isoformat(),
          'vg_metadata_images': len(metadata), 'identity_field_coverage': coverage,
          'missing_flickr_identity_counts': {key: len(value) for key, value in missing.items()},
          'metadata_sha256': hashlib.sha256(raw_metadata).hexdigest(),
          'gap_manifest_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
          'existing_coco_annotation_candidates': sorted(candidates.values(), key=lambda value: value['path_alias']),
          'directories_visited': len(visited), 'pending_directories': len(pending),
          'exhaustive_server_search': False, 'reference_text_used': False,
          'image_pixels_read': False, 'training_manifest_modified': False, 'benchmark_test_scored': False,
          'remaining': ['Extract image-only COCO/Flickr URL identities from valid existing annotation sources',
                        'Complete exact/perceptual image duplicate audit before final clean training release']}
(root / 'reports/CAPTION_IDENTITY_SOURCE_INVENTORY.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report))
