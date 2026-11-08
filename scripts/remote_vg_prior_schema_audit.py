"""CPU-only schema/source audit of the author prior, not a fresh training prior."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import signal
import subprocess
import numpy as np
import torch
torch.set_num_threads(1)
signal.alarm(90)
root = Path.home() / 'rsg-former'
vendor = root / 'vendor/salience_sgg'
path = vendor / 'pretrained/vg_prior_static.pt'
if not path.exists():
    # Discover the pinned local asset by name within this vendor, never scan shared projects.
    paths = list(vendor.rglob('vg_prior_static.pt'))
    assert len(paths) == 1
    path = paths[0]
sha = hashlib.sha256(path.read_bytes()).hexdigest()
assert sha == 'f4c84b9c0ad294aefbc10bb6112a0f4a6008d80e3203e8a34a6309c150421bcf'
# This is a known public, hash-locked author asset. It is read only for format audit.
value = torch.load(path, map_location='cpu', weights_only=False)
array = np.asarray(value)
assert array.ndim == 3 and np.isfinite(array).all() and (array >= 0).all()
matches = []
for file in vendor.rglob('*.py'):
    text = file.read_text(errors='replace')
    if any(token in text for token in ('fg_matrix', 'bg_matrix', 'get_statistics', 'vg_prior_static')):
        matches.append({'path': file.relative_to(vendor).as_posix(),
                        'sha256': hashlib.sha256(file.read_bytes()).hexdigest(),
                        'relevant_lines': [line.strip()[:200] for line in text.splitlines()
                                           if any(token in line for token in ('fg_matrix', 'bg_matrix', 'get_statistics', 'vg_prior_static'))]})
out = {'status': 'AUDITED_SCHEMA_ONLY', 'utc': datetime.now(timezone.utc).isoformat(),
       'author_asset_sha256': sha, 'python_type': type(value).__module__ + '.' + type(value).__name__,
       'shape': list(array.shape), 'dtype': str(array.dtype),
       'sum_all_entries': float(array.sum()), 'integer_valued': bool(np.equal(array, np.floor(array)).all()),
       'last_axis_sums': array.sum(axis=(0, 1)).tolist(), 'relevant_source': matches,
       'cuda_used': False, 'model_initialized': False, 'test_scored': False,
       'used_as_fresh_training_prior': False, 'clean_prior_generated': False,
       'remaining': ['Recover or specify source-equivalent count construction/class indexing/background channel',
                     'Recompute from finalized clean TRAIN IDs only; preserve separate provenance and source hash',
                     'Never reuse the author prior in fresh caption-clean SGG training']}
text = json.dumps(out, indent=2)
(root / 'reports/VG_PRIOR_SCHEMA_AUDIT.json').write_text(text)
print(text)
