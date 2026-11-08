"""Verify repeat-safe private scorer launchers with invented captions only."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import signal
import subprocess
import sys
import time

root = Path.home() / 'rsg-former'
prepared = json.loads((root / 'reports/CAPTION_JAVA_PREPARATION.json').read_text())
java = Path(prepared['java_path_alias'].replace('${REMOTE_HOME}', str(Path.home())))
assert hashlib.sha256(java.read_bytes()).hexdigest() == prepared['java_executable_sha256']
scratch = root / 'artifacts/caption_readiness/scorer_runtime_fixtures' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
scratch.mkdir(parents=True, exist_ok=False)
child = r'''
from pathlib import Path
import hashlib, importlib, json, os, sys
import numpy as np
root, scratch = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(root / 'src'))
os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
from rsgformer.caption_metric_runtime import JAVA8_OPENS, java8_spice_command, private_metric_runtime
command = ['java', '-jar', 'spice-1.0.jar']
for flag in sorted(JAVA8_OPENS):
    command += ['--add-opens', flag]
assert java8_spice_command(command)[0] == ['java', '-jar', 'spice-1.0.jar']
for bad in [command + ['--add-opens', sorted(JAVA8_OPENS)[0]],
            command + ['--add-opens', 'java.base/unknown=ALL-UNNAMED'],
            ['java', '-jar', 'spice-1.0.jar']]:
    try:
        java8_spice_command(bad)
    except ValueError:
        pass
    else:
        raise AssertionError('unexpected_launcher_flags_accepted')
spice = importlib.import_module('pycocoevalcap.spice.spice')
ptb = importlib.import_module('pycocoevalcap.tokenizer.ptbtokenizer')
assets = importlib.import_module('pycocoevalcap.spice.get_stanford_models')
before = (spice.subprocess.check_call, spice.TEMP_DIR, spice.CACHE_DIR, ptb.__file__, assets.urlretrieve)
refs = {1: [{'caption': 'A red ball rests on a table.'}, {'caption': 'A red ball is on a table.'}],
        2: [{'caption': 'Two blue birds fly above a tree.'}, {'caption': 'Two birds are flying above a tree.'}]}
preds = {1: [refs[1][0]], 2: [refs[2][0]]}
first_dir = scratch / 'first'
first_dir.mkdir()
with private_metric_runtime(root, first_dir) as runtime:
    tokenizer = runtime['modules']['PTBTokenizer'].PTBTokenizer()
    gts, res = tokenizer.tokenize(refs), tokenizer.tokenize(preds)
    observed = []
    for _ in range(2):
        aggregate, details = runtime['modules']['SPICE'].Spice().compute_score(gts, res)
        scores = [float(row['All']['f']) for row in details]
        assert np.isfinite(aggregate) and np.isfinite(scores).all() and len(scores) == 2
        observed.append([float(aggregate), *scores])
    assert np.allclose(observed[0], observed[1], rtol=0, atol=1e-12)
    assert len(runtime['spice_launcher_calls']) == 2
    assert all(len(call['removed_add_opens_pairs']) == 8 for call in runtime['spice_launcher_calls'])
    hashes = runtime['source_hashes']
assert before == (spice.subprocess.check_call, spice.TEMP_DIR, spice.CACHE_DIR, ptb.__file__, assets.urlretrieve)
second_dir = scratch / 'second'
second_dir.mkdir()
try:
    with private_metric_runtime(root, second_dir) as runtime:
        try:
            assets.urlretrieve('https://example.invalid/forbidden')
        except RuntimeError as error:
            assert str(error) == 'implicit_scorer_download_forbidden'
        else:
            raise AssertionError('implicit_download_allowed')
        try:
            with private_metric_runtime(root, second_dir):
                pass
        except RuntimeError as error:
            assert str(error) == 'nested_metric_runtime_forbidden'
        else:
            raise AssertionError('nested_runtime_allowed')
        raise ValueError('invented_metric_failure')
except ValueError as error:
    assert str(error) == 'invented_metric_failure'
assert before == (spice.subprocess.check_call, spice.TEMP_DIR, spice.CACHE_DIR, ptb.__file__, assets.urlretrieve)
for name, expected in hashes.items():
    from rsgformer.caption_metric_runtime import MODULES
    source = Path(importlib.import_module(MODULES[name]).__file__)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == expected
print(json.dumps({'status': 'PASS', 'repeated_actual_spice_calls': 2,
                  'repeat_metric_values_identical': True,
                  'launcher_flag_validation': 'PASS', 'implicit_download_guard': 'PASS',
                  'nested_runtime_guard': 'PASS', 'restoration_after_success_and_exception': 'PASS',
                  'scorer_source_hashes': hashes,
                  'shared_package_files_modified': False,
                  'references_or_benchmark_scores_read': False,
                  'benchmark_test_scored': False, 'model_weights_loaded': False,
                  'full_corpus_protocol_verified': False}))
'''
env = dict(os.environ)
env.update(PATH=str(java.parent) + os.pathsep + env.get('PATH', ''),
           CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
           _JAVA_OPTIONS='-Xmx2g -XX:ActiveProcessorCount=1 -XX:+UseSerialGC -XX:MaxMetaspaceSize=256m -XX:ReservedCodeCacheSize=128m')
started = time.monotonic()
process = subprocess.Popen([sys.executable, '-B', '-c', child, str(root), str(scratch)],
                           env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           text=True, start_new_session=True)
timed_out = False
try:
    output, diagnostic = process.communicate(timeout=120)
except subprocess.TimeoutExpired:
    timed_out = True
    os.killpg(process.pid, signal.SIGTERM)
    try:
        output, diagnostic = process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        output, diagnostic = process.communicate(timeout=5)
(scratch / 'stdout.log').write_text(output)
(scratch / 'stderr.log').write_text(diagnostic)
if process.returncode == 0:
    report = json.loads(output.splitlines()[-1])
else:
    report = {'status': 'FIXTURE_FAILED', 'exit_code': process.returncode,
              'timed_out': timed_out, 'diagnostic_tail': diagnostic[-3000:],
              'benchmark_test_scored': False}
report.update(utc=datetime.now(timezone.utc).isoformat(),
              elapsed_seconds=time.monotonic() - started,
              runtime_source_sha256=hashlib.sha256((root / 'src/rsgformer/caption_metric_runtime.py').read_bytes()).hexdigest(),
              fixture_source_sha256=hashlib.sha256((root / 'scripts/remote_caption_metric_runtime_probe.py').read_bytes()).hexdigest(),
              java_release=prepared['release_name'], heap_cap_gib=2,
              cpu_affinity_count=1, cuda_used=False, live_v2_modified=False,
              scratch_alias=str(scratch).replace(str(Path.home()), '${REMOTE_HOME}'),
              remaining=['Corpus metric wrapper and complete reference/prediction ID contract',
                         'Whole-corpus CIDEr document frequency and corpus BLEU/METEOR; SPICE batching verification',
                         'Freeze dataset, decoder, metric and selection protocols before locked evaluation'])
(root / 'reports/CAPTION_METRIC_RUNTIME_PROBE.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report))
if process.returncode != 0:
    sys.exit(1)
