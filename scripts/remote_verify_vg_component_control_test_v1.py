"""Read-only completion provenance verification; never runs model/evaluator/test."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import resource
import signal

resource.setrlimit(resource.RLIMIT_AS, (512*1024**2, 512*1024**2))
resource.setrlimit(resource.RLIMIT_CPU, (75, 75))
os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
signal.alarm(100)
root = Path.home()/'rsg-former'
art = root/'artifacts/vg_component_control_test_v1'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


lock_hash = sha(art/'preregistration.json')
assert lock_hash == '6b57ddbdbaf51c7d9307ae33f9766efc74cfe639cc51b98f9e36adf9dd659b77'
lock = json.loads((art/'preregistration.json').read_text())
assert sha(root/'reports/VG_COMPONENT_CONTROL_TEST_PREREGISTRATION.json') == lock_hash
for name, expected in lock['source_hashes'].items():
    assert sha(root/name) == expected, name
assert sha(root/lock['checkpoint']) == lock['checkpoint_sha256']
result = json.loads((art/'result.json').read_text())
assert result == json.loads((root/'reports/VG_COMPONENT_CONTROL_TEST_RESULT.json').read_text())
assert result['status'] == 'COMPONENT_TEST_COMPLETE' and result['test_passes'] == 1
assert result['checkpoint_sha256'] == lock['checkpoint_sha256']
assert result['test']['images'] == result['test']['observed_images'] == 26446
assert result['test']['empty_gt_images_excluded'] == 0
assert result['validation_gate']['status'] == 'PASS' and result['validation_gate']['images'] == 5000
assert result['validation_gate'] == json.loads((art/'validation_gate.json').read_text())
assert result['validation_gate'] == json.loads((root/'reports/VG_COMPONENT_CONTROL_VALIDATION_GATE.json').read_text())
access = json.loads((art/'test_access.json').read_text())
assert access == json.loads((root/'reports/VG_COMPONENT_TEST_ACCESS_LEDGER.json').read_text())
assert access['preregistration_sha256'] == lock_hash and access['checkpoint_sha256'] == lock['checkpoint_sha256']
for name, expected in result['artifact_hashes'].items():
    assert Path(name).name == name and sha(art/name) == expected, name
print(json.dumps({'utc': datetime.now(timezone.utc).isoformat(), 'status': 'PASS',
                  'preregistration_sha256': lock_hash, 'checkpoint_sha256': lock['checkpoint_sha256'],
                  'result_sha256': sha(art/'result.json'), 'source_files_verified': len(lock['source_hashes']),
                  'prediction_artifact_hashes_verified': result['artifact_hashes'],
                  'validation_gate_verified': True, 'test_ledger_verified': True,
                  'observed_test_images': 26446, 'new_test_inference_or_scoring': False}))
