"""CPU decoder/resource contracts on existing pilot pixels; no network/inference."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import signal
import sys
from collections import Counter
root = Path.home() / 'rsg-former'
sys.path.insert(0, str(root / 'src'))
signal.alarm(120)
from rsgformer.openimages_test_reader import OpenImagesTestReader
shared = Path.home() / 'shared-assets/datasets/openimages_test'
meta = json.loads((root / 'reports/OPENIMAGES_TEST_METADATA.json').read_text())
pilot = json.loads((root / 'reports/OPENIMAGES_TEST_PIXEL_PILOT.json').read_text())
assert pilot['status'] == 'complete' and len(pilot['records']) == 128
reader = OpenImagesTestReader(shared, manifest_sha256=meta['identity_manifest_sha256'],
                             known_pixel_records=pilot['records'])
assert len(reader) == 125436
orientation = Counter()
rotations = Counter()
for known in pilot['records']:
    item = reader.read(known['image_id'])
    assert list(item.image.size) == known['dimensions']
    assert item.provenance['sha256'] == known['sha256']
    assert not item.provenance['pixel_cache_written']
    orientation[str(item.provenance['exif_orientation'])] += 1
    rotations[str(item.provenance['metadata_rotation'])] += 1
    item.image.close()
checks = {}
for name, identity in [('non_test_identity', '../train/invalid'),
                       ('missing_pixel_network_disabled', next(i for i in reader.ids if i not in reader.expected_pixels))]:
    try: reader.read(identity)
    except (ValueError, RuntimeError) as error: checks[name] = str(error)
    else: raise AssertionError(name)
first = pilot['records'][0]
restricted = OpenImagesTestReader(shared, manifest_sha256=meta['identity_manifest_sha256'],
                                  single_image_bytes_cap=1)
try: restricted.read(first['image_id'])
except ValueError as error: checks['byte_cap'] = str(error)
else: raise AssertionError('byte_cap')
modified = dict(first, sha256='0' * 64)
restricted = OpenImagesTestReader(shared, manifest_sha256=meta['identity_manifest_sha256'],
                                  known_pixel_records=[modified])
try: restricted.read(first['image_id'])
except ValueError as error: checks['pixel_fingerprint'] = str(error)
else: raise AssertionError('pixel_fingerprint')
assert reader.downloaded_bytes == 0
out = {'status': 'PASS', 'utc': datetime.now(timezone.utc).isoformat(),
       'reader_sha256': hashlib.sha256((root / 'src/rsgformer/openimages_test_reader.py').read_bytes()).hexdigest(),
       'identity_manifest_sha256': meta['identity_manifest_sha256'], 'locked_test_id_count': len(reader),
       'existing_pilot_images_decoded': len(pilot['records']), 'exif_orientation_counts': dict(orientation),
       'pilot_metadata_rotation_counts': dict(rotations), 'negative_contracts': checks,
       'network_downloaded_bytes': reader.downloaded_bytes, 'disk_pixel_bytes_written': 0,
       'model_inference': False, 'test_scored': False, 'prompt_tuned': False,
       'full_pixel_coverage_audited': False, 'network_branch_tested': False,
       'remaining': ['Versioned method/prompt/decoder/orientation/evaluator locks before test inference',
                     'Bounded network branch and full-ID completeness/retry/output journal integration',
                     'Missing/corrupt images must block final complete-set scoring rather than silently shrink denominator']}
text = json.dumps(out, indent=2)
(root / 'reports/OPENIMAGES_TEST_READER_PROBE.json').write_text(text)
print(text)
