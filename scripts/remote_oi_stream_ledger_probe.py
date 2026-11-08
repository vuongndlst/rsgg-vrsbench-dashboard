"""Fault injection into pinned pilot ledger, isolated roots and invented RGB.

Uses the production script body, replacing only root and reader in memory.
No network, real pixel journal, benchmark references, GPU, or model access.
"""
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '1'
from pathlib import Path
from datetime import datetime, timezone
import ast
import contextlib
import fcntl
import hashlib
import io
import json
import shutil
import signal
import sqlite3
import sys
import time
import types
from PIL import Image

root = Path.home() / 'rsg-former'
os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
signal.alarm(120)
started = time.monotonic()
source_path = root / 'scripts/remote_oi_stream_fingerprint_pilot.py'
source = source_path.read_bytes()
source_sha = hashlib.sha256(source).hexdigest()
assert source_sha == '98d1082ea7681c8a1147f6cd6cfcf66d1015c537face2e3c7fa2dc59d02e53fb'
folder = root / 'artifacts/caption_readiness/oi_stream_ledger_fixture_v1'
assert not folder.exists(), 'preserve_existing_fixture_version'
assert shutil.disk_usage(root).free > 16 * 1024 ** 3 + 64 * 1024 ** 2
folder.mkdir()
tree = ast.parse(source)
changed = 0
for node in tree.body:
    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id=='root' for t in node.targets):
        assert ast.unparse(node.value) == "Path.home() / 'rsg-former'"
        node.value = ast.Name(id='fixture_root', ctx=ast.Load())
        changed += 1
assert changed == 1
ast.fix_missing_locations(tree)
code = compile(tree, 'isolated_pinned_stream_ledger', 'exec')
capsule = {'utc': datetime.now(timezone.utc).isoformat(), 'production_source_sha256': source_sha,
           'instrumentation': 'one AST root substitution and temporary fake reader module; ledger/control body unchanged',
           'cases': ['success and zero-network resume', 'crash with committed inflight reservation then recovery',
                     'partial byte failure with conservative charge', 'two-attempt guard',
                     'cumulative cap admission before request', 'OS lock prevents another writer',
                     'source drift refusal'],
           'network_allowed': False, 'invented_pixels_only': True, 'cpu_threads': 1,
           'timeout_seconds': 120, 'live_v2_modified': False}
(folder / 'preregistration.json').write_text(json.dumps(capsule, indent=2))

class InjectedCrash(BaseException):
    pass

results = {}
current = {}

class FakeReader:
    def __init__(self, *args, **kwargs):
        self.ids = tuple(f'{number:016x}' for number in range(20))
        self.downloaded_bytes = 0

    def read(self, identity):
        current['reads'].append(identity)
        lock_path = current['root'] / 'artifacts/caption_readiness/oi_stream_fingerprints_pilot_v1.lock'
        with lock_path.open('a') as other:
            try:
                fcntl.flock(other.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                current['lock_guard'] = True
            else:
                raise AssertionError('production_writer_lock_not_held')
        mode = current['mode']
        if mode == 'crash' and identity == self.ids[0]:
            raise InjectedCrash()
        self.downloaded_bytes += 100
        if mode == 'fail_first' and identity == self.ids[0]:
            raise ValueError('injected_partial_download_after100bytes')
        if mode == 'fail_all':
            raise ValueError('injected_failure_after100bytes')
        rgb = Image.new('RGB', (8, 16), (int(identity, 16), 40, 80))
        return types.SimpleNamespace(image=rgb, provenance={'sha256':hashlib.sha256(identity.encode()).hexdigest(),
                                     'pixel_cache_written':False,'loaded_from':'invented_fixture', 'bytes':100})

module_name = 'rsgformer.openimages_test_reader'
previous = sys.modules.get(module_name)
fake_module = types.ModuleType(module_name)
fake_module.OpenImagesTestReader = FakeReader
sys.modules[module_name] = fake_module

def case_root(name):
    target = folder / name
    (target / 'src/rsgformer').mkdir(parents=True)
    (target / 'scripts').mkdir()
    (target / 'reports').mkdir()
    shutil.copyfile(root / 'src/rsgformer/openimages_test_reader.py', target / 'src/rsgformer/openimages_test_reader.py')
    (target / 'scripts/remote_oi_stream_fingerprint_pilot.py').write_bytes(source)
    return target

def execute(target, mode):
    current.update(root=target, mode=mode, reads=[], lock_guard=False)
    with contextlib.redirect_stdout(io.StringIO()):
        exec(code, {'fixture_root': target, '__name__': '__main__'})
    report = json.loads((target / 'reports/OI_STREAM_FINGERPRINT_PILOT.json').read_text())
    return report, list(current['reads'])

def db_for(target):
    return sqlite3.connect(target / 'artifacts/caption_readiness/oi_stream_fingerprints_pilot_v1.sqlite')

try:
    target = case_root('success_resume')
    report, reads = execute(target, 'success')
    assert len(reads) == 20 and report['indexed_images'] == 20 and report['persistent_network_charge_bytes'] == 2000
    assert current['lock_guard']
    resumed, reads = execute(target, 'success')
    assert not reads and resumed['batch_reader_counted_network_bytes'] == 0
    assert resumed['fingerprint_rows_sha256'] == report['fingerprint_rows_sha256']
    results['success_resume_and_writer_lock'] = 'PASS'
    target = case_root('crash_recovery')
    try:
        execute(target, 'crash')
    except InjectedCrash:
        pass
    else:
        raise AssertionError('crash_not_injected')
    with db_for(target) as db:
        assert db.execute('SELECT state,charged_bytes FROM attempts').fetchall() == [('inflight', 8*1024**2)]
        assert db.execute('SELECT COUNT(*) FROM pixels').fetchone()[0] == 0
    recovered, reads = execute(target, 'success')
    assert len(reads) == 20 and recovered['persistent_network_charge_bytes'] == 8*1024**2 + 2000
    with db_for(target) as db:
        assert db.execute("SELECT COUNT(*) FROM attempts WHERE state='interrupted'").fetchone()[0] == 1
    results['crash_retains_reservation_and_recovers'] = 'PASS'
    target = case_root('failure_retry_guard')
    first, reads = execute(target, 'fail_first')
    assert first['indexed_images'] == 19 and first['persistent_network_charge_bytes'] == 8*1024**2 + 1900
    assert first['batch_reader_counted_network_bytes'] == 2000
    second, reads = execute(target, 'fail_first')
    assert len(reads) == 1 and second['persistent_network_charge_bytes'] == 16*1024**2 + 1900
    third, reads = execute(target, 'success')
    assert not reads and third['indexed_images'] == 19
    results['partial_failure_charge_and_two_attempt_guard'] = 'PASS'
    target = case_root('cap_admission')
    report, reads = execute(target, 'fail_all')
    assert len(reads) == 4 and report['persistent_network_charge_bytes'] == 32*1024**2
    assert report['stop_reason'] == 'persistent_network_charge_cap'
    resumed, reads = execute(target, 'success')
    assert not reads and resumed['persistent_network_charge_bytes'] == 32*1024**2
    results['persistent_cap_stops_before_request'] = 'PASS'
    target = folder / 'success_resume'
    with db_for(target) as db:
        db.execute('UPDATE provenance SET value=? WHERE key=?', ('deliberately_changed_fixture_source', 'source'))
    try:
        execute(target, 'success')
    except AssertionError as error:
        assert str(error) == 'pilot_source_changed_new_version_required'
    else:
        raise AssertionError('source_drift_not_rejected')
    assert not current['reads']
    results['source_drift_refusal_before_network'] = 'PASS'
    outcome = {'status':'PASS', 'utc':datetime.now(timezone.utc).isoformat(),
               'production_source_sha256':source_sha,
               'fixture_source_sha256':hashlib.sha256((root / 'scripts/remote_oi_stream_ledger_probe.py').read_bytes()).hexdigest(),
               'cases':results, 'elapsed_seconds':time.monotonic()-started,
               'network_downloaded_bytes':0, 'real_pixel_journals_modified':False,
               'benchmark_test_scored':False, 'reference_captions_read':False,
               'model_or_metric_run':False, 'live_v2_modified':False,
               'full_streaming_registered':False,
               'remaining':['Register full protected TEST streaming budget/version and continuation policy',
                            'Join validated streaming fingerprints with VG exact/perceptual audit']}
    (folder / 'outcome.json').write_text(json.dumps(outcome, indent=2))
    (root / 'reports/OI_STREAM_LEDGER_PROBE.json').write_text(json.dumps(outcome,indent=2))
    print(json.dumps(outcome))
finally:
    if previous is None:
        sys.modules.pop(module_name, None)
    else:
        sys.modules[module_name] = previous
    signal.alarm(0)
