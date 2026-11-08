"""Bounded, read-only CPU cache loading profile; no model, GPU or benchmark scores."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, os, resource, shutil, signal, time

os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
os.nice(19)
signal.alarm(60)
resource.setrlimit(resource.RLIMIT_AS, (4*1024**3, 4*1024**3))
root = Path.home() / 'rsg-former'
assert shutil.disk_usage(root).free > 16*1024**3 + 1024**2
folder = root / 'artifacts/caption_readiness/vg_cache_io_speed_probe_v2'
assert not folder.exists(), 'preserve_completed_profile_no_rerun'
manifest_path = root / 'artifacts/vg_method_v2/features/manifest.json'
manifest_raw = manifest_path.read_bytes()
manifest = json.loads(manifest_raw)
assert manifest['status'] == 'complete' and manifest['cache_dtype'] == 'float32'
lock = json.loads((root / 'reports/VG_V2_PREREGISTRATION.json').read_text())
assert manifest['preregistration_source_hashes'] == lock['source_hashes']
for name, expected in lock['source_hashes'].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected
entries = manifest['splits']['train'][:2]
assert len(entries) == 2 and sum(e['images'] for e in entries) <= 128
protocol = {'utc': datetime.now(timezone.utc).isoformat(),
            'script_sha256': hashlib.sha256((root / 'scripts/remote_vg_cache_io_speed_probe.py').read_bytes()).hexdigest(),
            'manifest_sha256': hashlib.sha256(manifest_raw).hexdigest(),
            'source_hashes': lock['source_hashes'],
            'selection': 'first two existing train feature shards, fixed before timing; two sequential reads',
            'preflight_repair': 'v1 stopped at source-read cap assertion before artifact creation or shard reads; v2 reduces sample without raising caps',
            'shards': entries, 'resource': {'cpu_affinity_count': 1, 'threads': 1, 'alarm_seconds': 60,
                                         'address_space_bytes': 4*1024**3, 'peak_rss_gate_bytes': 768*1024**2,
                                         'source_read_bytes_cap': 160*1024**2, 'disk_reserve_bytes': 16*1024**3},
            'cuda_used': False, 'model_loaded': False, 'image_pixels_read': False,
            'validation_or_test_accessed': False, 'live_v2_modified': False}
assert 3*sum(e['bytes'] for e in entries) <= protocol['resource']['source_read_bytes_cap']
folder.mkdir()
raw = json.dumps(protocol, indent=2).encode()
(folder / 'preregistration.json').write_bytes(raw)
import torch
torch.set_num_threads(1)
cache = manifest_path.parent
for entry in entries:
    assert Path(entry['path']).name == entry['path']
    p = cache / entry['path']
    assert p.stat().st_size == entry['bytes']
    assert hashlib.sha256(p.read_bytes()).hexdigest() == entry['sha256']
runs = []
for repeat in range(2):
    timings = []
    for entry in entries:
        start = time.perf_counter()
        chunk = torch.load(cache / entry['path'], map_location='cpu', weights_only=True)
        duration = time.perf_counter() - start
        assert len(chunk) == entry['images']
        assert [item['image_id'] for item in chunk] == entry['image_ids']
        assert all(item[k].dtype == torch.float32 for item in chunk for k in ('nodes','boxes','logits','visual'))
        timings.append({'path': entry['path'], 'images': len(chunk), 'seconds': duration,
                        'bytes': entry['bytes']})
        del chunk
    runs.append({'repeat': repeat, 'shards': timings, 'images': sum(t['images'] for t in timings),
                 'total_load_seconds': sum(t['seconds'] for t in timings)})
for run in runs:
    run['load_seconds_per_image'] = run['total_load_seconds']/run['images']
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
assert peak <= protocol['resource']['peak_rss_gate_bytes'], 'cpu_rss_gate_exceeded'
for name, expected in lock['source_hashes'].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected
result = {'utc': datetime.now(timezone.utc).isoformat(), 'status': 'CACHE_IO_TIMING_COMPLETE',
          'preregistration_sha256': hashlib.sha256(raw).hexdigest(), 'protocol': protocol,
          'runs': runs, 'peak_cpu_rss_bytes': peak,
          'total_cache_bytes_read_including_hash_verification': 3*sum(e['bytes'] for e in entries),
          'cuda_used': False, 'model_loaded': False, 'validation_or_test_accessed': False,
          'live_v2_modified': False, 'cache_modified': False,
          'limits': 'Small fixed train shard sample; existing page cache not flushed; excludes model/loss/transfer/GT/validation timings and full-epoch I/O contention. No speedup factor inferred.'}
(folder / 'outcome.json').write_text(json.dumps(result, indent=2))
(root / 'reports/VG_CACHE_IO_SPEED_PROBE.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result))
