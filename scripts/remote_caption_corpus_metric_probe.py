"""Native COCOEvalCap oracle vs corpus wrapper on invented captions only."""
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
scratch = root / 'artifacts/caption_readiness/corpus_metric_fixtures' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
scratch.mkdir(parents=True, exist_ok=False)
child = r'''
from pathlib import Path
from types import SimpleNamespace
import hashlib, importlib, json, math, os, sys
root, scratch = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(root / 'src'))
os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
from rsgformer.caption_metric_runtime import private_metric_runtime
from rsgformer.caption_metrics import compute_caption_metrics, validate_caption_units
refs = {'z:1': ['A red ball rests on a table.', 'A red ball is on a table.'],
        'a:1': ['Two blue birds fly above a tree.', 'Two birds are flying above a tree.'],
        'm:2': ['A woman holds a green umbrella.', 'A person holds an umbrella.'],
        'b:3': ['Three dogs run beside a fence.', 'Dogs are running near a fence.']}
preds = {'z:1': refs['z:1'][0], 'a:1': 'A blue bird rests on a table.',
         'm:2': 'A woman holds a red umbrella.', 'b:3': 'Three dogs run beside a fence.'}
for bad_refs, bad_preds in [(refs, {key: value for key, value in preds.items() if key != 'z:1'}),
                            ({**refs, 'z:1': []}, preds), ({**refs, 'z:1': ['']}, preds),
                            (refs, {**preds, 'z:1': ['multiple', 'captions']})]:
    try:
        validate_caption_units(bad_refs, bad_preds)
    except ValueError:
        pass
    else:
        raise AssertionError('invalid_caption_input_accepted')
runtime_dir = scratch / 'runtime'
runtime_dir.mkdir()
with private_metric_runtime(root, runtime_dir) as runtime:
    measured = compute_caption_metrics(refs, preds, runtime, spice_batch_units=2)
    ordered = measured['ordered_unit_ids']
    fake_coco = SimpleNamespace(
        getImgIds=lambda: list(range(1, 5)),
        imgToAnns={index: [{'caption': caption} for caption in refs[identity]]
                   for index, identity in enumerate(ordered, 1)})
    fake_res = SimpleNamespace(imgToAnns={index: [{'caption': preds[identity]}]
                                        for index, identity in enumerate(ordered, 1)})
    native = importlib.import_module('pycocoevalcap.eval')
    oracle = native.COCOEvalCap(fake_coco, fake_res)
    oracle.evaluate()
    assert set(measured['aggregate']) == set(oracle.eval)
    max_aggregate_error, max_unit_error = 0.0, 0.0
    for name, score in measured['aggregate'].items():
        expected = float(oracle.eval[name])
        assert math.isfinite(expected) and math.isclose(score, expected, rel_tol=1e-10, abs_tol=1e-12)
        max_aggregate_error = max(max_aggregate_error, abs(score - expected))
    for index, identity in enumerate(ordered, 1):
        for name, score in measured['per_unit'][identity].items():
            value = oracle.imgToEval[index][name]
            expected = float(value['All']['f'] if name == 'SPICE' else value)
            assert math.isclose(score, expected, rel_tol=1e-10, abs_tol=1e-12)
            max_unit_error = max(max_unit_error, abs(score - expected))
    assert [batch['units'] for batch in measured['protocol']['spice_batches']] == [2, 2]
    assert len(runtime['spice_launcher_calls']) == 3
    # The wrapper reports native corpus BLEU rather than mean sentence BLEU.
    sentence_mean = sum(row['Bleu_4'] for row in measured['per_unit'].values()) / 4
    assert not math.isclose(sentence_mean, measured['aggregate']['Bleu_4'], abs_tol=1e-5)
    oracle_sha = hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest()
print(json.dumps({'status': 'PASS', 'fixture_units': 4, 'fixture_references': 8,
                  'oracle': 'actual installed pycocoevalcap.COCOEvalCap complete corpus',
                  'oracle_source_sha256': oracle_sha,
                  'metric_names': sorted(measured['aggregate']),
                  'max_aggregate_absolute_error': max_aggregate_error,
                  'max_per_unit_absolute_error': max_unit_error,
                  'spice_batch_vs_full_corpus': 'PASS',
                  'corpus_bleu_not_mean_sentence_bleu': 'PASS',
                  'missing_unit_empty_reference_multiple_prediction_guards': 'PASS',
                  'sorted_composite_id_alignment': 'PASS',
                  'metric_protocol': measured['protocol'],
                  'references_or_benchmark_scores_read': False,
                  'benchmark_test_scored': False, 'model_weights_loaded': False,
                  'shared_package_files_modified': False,
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
    report = {'status': 'FIXTURE_FAILED', 'exit_code': process.returncode, 'timed_out': timed_out,
              'diagnostic_tail': diagnostic[-3000:], 'benchmark_test_scored': False}
report.update(utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic() - started,
              source_hashes={name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in
                             ['src/rsgformer/caption_metrics.py', 'src/rsgformer/caption_metric_runtime.py',
                              'scripts/remote_caption_corpus_metric_probe.py']},
              java_release=prepared['release_name'], heap_cap_gib=2, cpu_affinity_count=1,
              cuda_used=False, live_v2_modified=False,
              scratch_alias=str(scratch).replace(str(Path.home()), '${REMOTE_HOME}'),
              remaining=['Measured corpus tokenization/memory/time and SPICE batch throughput',
                         'Full scorer dependencies/runtime hashes and corpus options lock',
                         'Complete data/crop/clean model/decoder/prompt/metric preregistration before test'])
(root / 'reports/CAPTION_CORPUS_METRIC_PROBE.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report))
if process.returncode != 0:
    sys.exit(1)
