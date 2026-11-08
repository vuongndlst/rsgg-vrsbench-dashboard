"""Invented-score image-cluster statistical contracts; no benchmark data."""
import os
for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[name] = '1'
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import signal
import sys

signal.alarm(30)
os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
root = Path.home() / 'rsg-former'
sys.path.insert(0, str(root / 'src'))
from rsgformer.caption_statistics import paired_image_bootstrap
kwargs = {'repetitions': 1000, 'seed': 6723}  # Fixture seed only, not a benchmark registration.
left = {'a': .1, 'b': .3, 'c': .5}
parents = {'a': 'image_a', 'b': 'image_b', 'c': 'image_c'}
zero = paired_image_bootstrap(left, left, parents, **kwargs)
assert zero['paired_delta'] == 0 and zero['ci95_percentile'] == [0., 0.]
right = {key: value + 1 for key, value in left.items()}
constant = paired_image_bootstrap(left, right, parents, **kwargs)
assert abs(constant['paired_delta'] - 1) < 1e-12
assert all(abs(value - 1) < 1e-12 for value in constant['ci95_percentile'])
region_left = {'r1': 0., 'r2': 0., 'r3': 0.}
region_right = {'r1': 1., 'r2': 0., 'r3': 0.}
region_parents = {'r1': 'image_a', 'r2': 'image_b', 'r3': 'image_b'}
cluster = paired_image_bootstrap(region_left, region_right, region_parents, **kwargs)
assert cluster['paired_delta'] == 1 / 3  # Not the incorrectly reweighted image mean 1/2.
assert cluster['ci95_percentile'] == [0., 1.]
assert cluster['parent_image_count'] == 2 and cluster['unit_count'] == 3
reordered = paired_image_bootstrap(dict(reversed(list(region_left.items()))),
                                  dict(reversed(list(region_right.items()))), region_parents, **kwargs)
assert reordered == cluster
checks = {}
for name, a, b, p in [('missing_unit', left, {'a': 1.}, parents),
                     ('nonfinite', left, dict(right, a=float('nan')), parents),
                     ('one_parent_image', left, right, {key: 'only' for key in left}),
                     ('missing_parent', left, right, {'a': 'image_a'})]:
    try:
        paired_image_bootstrap(a, b, p, **kwargs)
    except ValueError as error:
        checks[name] = str(error)
    else:
        raise AssertionError(name)
report = {'status': 'PASS', 'utc': datetime.now(timezone.utc).isoformat(),
          'source_sha256': hashlib.sha256((root / 'src/rsgformer/caption_statistics.py').read_bytes()).hexdigest(),
          'identical_arms_zero_interval': True, 'constant_gain_exact_interval': True,
          'unequal_region_cluster_mean_preserves_unit_estimand': True,
          'input_order_invariance': True, 'negative_contracts': checks,
          'fixture_seed_is_not_benchmark_registration': True,
          'references_or_benchmark_scores_read': False, 'model_or_metric_run': False,
          'benchmark_test_scored': False, 'live_v2_modified': False,
          'remaining': ['Preregister caption CI seed/repetitions and parent-image/unit membership before test',
                        'Use fixed-corpus per-unit CIDEr/SPICE scores; report conditional IDF scope',
                        'Do not substitute averaged sentence BLEU/METEOR for native corpus metrics',
                        'Freeze VG region coverage/no-match policy independently before scoring']}
(root / 'reports/CAPTION_STATISTICS_PROBE.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report))
