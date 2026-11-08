"""Bounded test-only CVDF network/decoder proof; no model, references or metrics."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import signal
import struct
import sys

signal.alarm(90)
os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
root = Path.home() / 'rsg-former'
sys.path.insert(0, str(root / 'src'))
source = root / 'src/rsgformer/openimages_test_reader.py'
reader_sha = hashlib.sha256(source.read_bytes()).hexdigest()
assert reader_sha == '0bf8d41c541109e11c89e3e7dbaf47b38234a87a995baf2e5f2f2e0b16d307f6'
from rsgformer.openimages_test_reader import OpenImagesTestReader
shared = Path.home() / 'shared-assets/datasets/openimages_test'
manifest_sha = '6eb586b40bd68245e20205d896a5ae49700b72d61783b0e5ddc80a50e2242c87'
reader = OpenImagesTestReader(shared, manifest_sha256=manifest_sha, allow_network=True,
                             network_bytes_cap=8 * 1024 ** 2)
ids = [identity for identity in reader.ids if not (shared / 'images' / (identity + '.jpg')).exists()][:3]
assert len(ids) == 3
records = []
for identity in ids:
    item = reader.read(identity)
    record = dict(item.provenance)
    assert record['loaded_from'] == 'in_memory_public_cvdf_download'
    assert not record['pixel_cache_written'] and not (shared / 'images' / (identity + '.jpg')).exists()
    record['decoded_rgb_sha256'] = hashlib.sha256(struct.pack('>II', *item.image.size) + item.image.tobytes()).hexdigest()
    records.append(record)
    item.image.close()
assert sum(record['bytes'] for record in records) == reader.downloaded_bytes <= 8 * 1024 ** 2
restricted = OpenImagesTestReader(shared, manifest_sha256=manifest_sha, allow_network=True, network_bytes_cap=0)
try:
    restricted.read(ids[0])
except RuntimeError as error:
    assert str(error) == 'network_byte_budget_exhausted'
else:
    raise AssertionError('zero_network_budget_guard')
assert restricted.downloaded_bytes == 0
artifact = root / 'artifacts/caption_readiness/oi_network_probe_pixel_fingerprints.json'
raw = json.dumps({'reader_sha256': reader_sha, 'identity_manifest_sha256': manifest_sha,
                  'records': records}, separators=(',', ':')).encode()
assert not artifact.exists(), 'preserve_existing_network_probe_evidence'
artifact.write_bytes(raw)
report = {'status': 'PASS', 'utc': datetime.now(timezone.utc).isoformat(),
          'reader_sha256': reader_sha, 'identity_manifest_sha256': manifest_sha,
          'network_branch_tested': True, 'network_images_fully_decoded': len(records),
          'selection': 'first three sorted locked TEST IDs absent from existing cache; no references/metrics',
          'network_downloaded_bytes': reader.downloaded_bytes, 'network_byte_cap': 8 * 1024 ** 2,
          'zero_budget_guard': 'PASS_before_network', 'cpu_affinity_count': len(os.sched_getaffinity(0)),
          'evidence_artifact_sha256': hashlib.sha256(raw).hexdigest(),
          'disk_pixel_bytes_written': 0, 'reference_captions_read': False,
          'model_inference': False, 'test_scored': False, 'prompt_tuned': False,
          'oi_train_or_val_accessed': False, 'live_v2_modified': False,
          'full_pixel_coverage_audited': False,
          'remaining': ['Full-ID retry/completeness and cumulative byte/output journal for streaming',
                        'Protected pixel duplicate audit and locked orientation/decoder/exclusion policy',
                        'Freeze clean method/prompt/metric protocols before any TEST model inference']}
(root / 'reports/OPENIMAGES_NETWORK_READER_PROBE.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report))
