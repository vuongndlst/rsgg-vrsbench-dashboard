"""Actual pycocoevalcap synthetic fixtures in a private, bounded CPU subprocess."""
from pathlib import Path
from datetime import datetime, timezone
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
assert root.resolve() in java.resolve().parents
assert hashlib.sha256(java.read_bytes()).hexdigest() == prepared['java_executable_sha256']
stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
scratch = root / 'artifacts/caption_readiness/scorer_fixtures' / stamp
scratch.mkdir(parents=True, exist_ok=False)
assert (root / 'artifacts/caption_readiness').resolve() in scratch.resolve().parents

code = r'''
from pathlib import Path
import contextlib, hashlib, importlib, io, json, os, sys, urllib.request
import numpy as np
root, scratch = Path(sys.argv[1]), Path(sys.argv[2])
os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
preflight = json.loads((root / 'reports/CAPTION_RUNTIME_PREFLIGHT.json').read_text())
names = {'PTBTokenizer': 'pycocoevalcap.tokenizer.ptbtokenizer',
         'METEOR': 'pycocoevalcap.meteor.meteor', 'SPICE': 'pycocoevalcap.spice.spice',
         'BLEU': 'pycocoevalcap.bleu.bleu', 'CIDEr': 'pycocoevalcap.cider.cider',
         'ROUGE_L': 'pycocoevalcap.rouge.rouge'}
modules, hashes = {}, {}
for name, dotted in names.items():
    module = importlib.import_module(dotted)
    source = Path(module.__file__)
    hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
    assert hashes[name] == preflight['scorers'][name]['source_sha256']
    for jar in preflight['scorers'][name]['jars']:
        jar_path = source.parent / jar['relative_path']
        assert hashlib.sha256(jar_path.read_bytes()).hexdigest() == jar['sha256']
    modules[name] = module

def forbidden_download(*args, **kwargs):
    raise RuntimeError('implicit_scorer_download_forbidden')
assets = importlib.import_module('pycocoevalcap.spice.get_stanford_models')
assets.urlretrieve = forbidden_download
spice = modules['SPICE']
spice.TEMP_DIR, spice.CACHE_DIR = str(scratch / 'spice_tmp'), str(scratch / 'spice_cache')
# The installed wrapper has Java9+ module flags, but the original SPICE jar
# requires the Java8 Nashorn engine. Remove only those incompatible launcher
# flags in this private process; retain the actual metric jar/algorithm.
original_check_call = spice.subprocess.check_call
removed_module_flags = []
allowed_modules = {'java.base/' + name + '=ALL-UNNAMED' for name in
                   ['java.lang', 'java.util', 'java.util.concurrent', 'java.io',
                    'java.math', 'java.net', 'java.text', 'java.lang.reflect']}
def java8_spice_check_call(command, *args, **kwargs):
    assert command[0] == 'java' and 'spice-1.0.jar' in command
    adjusted, index = [], 0
    while index < len(command):
        if command[index] == '--add-opens':
            assert command[index + 1] in allowed_modules
            removed_module_flags.append(command[index + 1])
            index += 2
        else:
            adjusted.append(command[index])
            index += 1
    assert set(removed_module_flags) == allowed_modules and len(removed_module_flags) == 8
    return original_check_call(adjusted, *args, **kwargs)
spice.subprocess.check_call = java8_spice_check_call
ptb = modules['PTBTokenizer']
ptb_source = Path(ptb.__file__)
ptb_scratch = scratch / 'ptb'
ptb_scratch.mkdir()
(ptb_scratch / ptb.STANFORD_CORENLP_3_4_1_JAR).symlink_to(ptb_source.parent / ptb.STANFORD_CORENLP_3_4_1_JAR)
ptb.__file__ = str(ptb_scratch / ptb_source.name)
# Only temporary-file locations change; actual installed scorers and original jars execute.
raw_refs = {1: [{'caption': 'A red ball rests on a table.'}, {'caption': 'A red ball is on a table.'}],
            2: [{'caption': 'Two blue birds fly above a tree.'}, {'caption': 'Two birds are flying above a tree.'}]}
raw_preds = {1: [raw_refs[1][0]], 2: [raw_refs[2][0]]}
tokenizer = ptb.PTBTokenizer()
references, predictions = tokenizer.tokenize(raw_refs), tokenizer.tokenize(raw_preds)
assert references[1][0] == 'a red ball rests on a table'
assert set(references) == set(predictions) == {1, 2}
assert len(references[1]) == len(references[2]) == 2
assert len(predictions[1]) == len(predictions[2]) == 1
results = {'PTBTokenizer': {'status': 'PASS', 'lowercase_punctuation_reference_count_contract': 'PASS'}}
for name, scorer in [('BLEU', modules['BLEU'].Bleu(4)), ('CIDEr', modules['CIDEr'].Cider()),
                     ('ROUGE_L', modules['ROUGE_L'].Rouge())]:
    with contextlib.redirect_stdout(io.StringIO()):
        score, per_image = scorer.compute_score(references, predictions)
    assert np.isfinite(np.asarray(score)).all() and np.isfinite(np.asarray(per_image)).all()
    assert np.asarray(per_image).shape[-1] == 2
    results[name] = {'status': 'PASS', 'aggregate_and_image_scores_finite': True}
meteor = modules['METEOR'].Meteor()
try:
    score, per_image = meteor.compute_score(references, predictions)
    assert np.isfinite(score) and np.isfinite(per_image).all() and len(per_image) == 2
    results['METEOR'] = {'status': 'PASS', 'aggregate_and_image_scores_finite': True}
finally:
    if meteor.lock.locked():
        meteor.lock.release()
    del meteor  # Native destructor terminates only this fixture's Meteor child.
score, per_image = spice.Spice().compute_score(references, predictions)
assert np.isfinite(score) and len(per_image) == 2
assert all(np.isfinite(item['All']['f']) for item in per_image)
results['SPICE'] = {'status': 'PASS', 'aggregate_and_image_all_finite': True,
                    'undefined_absent_subcategories_allowed': True}
scratch_bytes = sum(path.stat().st_size for path in scratch.rglob('*') if path.is_file() and not path.is_symlink())
assert scratch_bytes < 64 * 1024 ** 2
print(json.dumps({'status': 'PASS', 'scorers': results, 'scorer_source_hashes': hashes,
                  'java8_launcher_adapter': {'removed_add_opens_pairs': removed_module_flags,
                                              'metric_jar_and_algorithm_modified': False},
                  'scratch_regular_file_bytes': scratch_bytes, 'model_weights_loaded': False,
                  'benchmark_validation_scored': False, 'benchmark_test_scored': False,
                  'shared_package_files_modified': False, 'implicit_download_requested': False,
                  'score_values_are_not_benchmark_results': True}))
'''
env = dict(os.environ)
env.update(PATH=str(java.parent) + os.pathsep + env.get('PATH', ''),
           CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
           _JAVA_OPTIONS='-Xmx2g -XX:ActiveProcessorCount=1 -XX:+UseSerialGC -XX:MaxMetaspaceSize=256m -XX:ReservedCodeCacheSize=128m')
started = time.monotonic()
process = subprocess.Popen([sys.executable, '-B', '-c', code, str(root), str(scratch)],
                           env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           text=True, start_new_session=True)
timed_out = False
try:
    output, diagnostic = process.communicate(timeout=150)
except subprocess.TimeoutExpired:
    timed_out = True
    # This process group was created here solely for this fixture; no other jobs are targeted.
    os.killpg(process.pid, signal.SIGTERM)
    try:
        output, diagnostic = process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        output, diagnostic = process.communicate(timeout=5)
    diagnostic += '\nfixture timeout; exact fixture process group terminated'
(scratch / 'stdout.log').write_text(output)
(scratch / 'stderr.log').write_text(diagnostic)
if process.returncode == 0:
    # Java SPICE prints timing information before the fixture's final JSON line.
    report = json.loads(output.splitlines()[-1])
    assert report['status'] == 'PASS' and len(report['scorers']) == 6
    report['child_stdout_chatter_lines'] = len(output.splitlines()) - 1
else:
    report = {'status': 'FIXTURE_FAILED', 'exit_code': process.returncode, 'timed_out': timed_out,
              'diagnostic_tail': diagnostic[-3000:], 'benchmark_test_scored': False}
report.update(utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic() - started,
              java_release=prepared['release_name'], java_archive_sha256=prepared['archive_sha256'],
              heap_cap_gib=2, java_active_processors=1, cuda_used=False,
              private_scratch_alias=str(scratch).replace(str(Path.home()), '${REMOTE_HOME}'),
              live_v2_modified=False)
(root / 'reports/CAPTION_JAVA_METRIC_PROBE.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report))
if process.returncode != 0:
    sys.exit(1)
