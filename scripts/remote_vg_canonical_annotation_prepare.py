"""Download only the official RelTR VG annotation archive for release parity audit.

Never extracts images or changes the active benchmark release. The downloader is
our own child process, bounded in time, bytes, and disk reserve.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import subprocess
import sys
import time
import zipfile

root = Path.home() / 'rsg-former'
dest = root / 'data/vg_canonical_audit'
dest.mkdir(parents=True, exist_ok=True)
report = root / 'reports/VG_CANONICAL_ANNOTATION_PREPARATION.json'
archive = dest / 'reltr_vg_annotations.zip'
ident = '1aGwEu392DiECGdvwaYr-LgqGLmWhn8yD'
out = {'schema_version': 1, 'source_readme': 'https://github.com/yrcong/RelTR/blob/main/data/README.md',
       'public_file_id': ident, 'mode': 'annotation_archive_only_no_extraction_no_scoring',
       'active_v2_release_modified': False, 'test_performance_scored': False,
       'time_limit_seconds': 600, 'archive_limit_bytes': 512 * 2**20}

def emit(status, **values):
    out.update(status=status, updated_utc=datetime.now(timezone.utc).isoformat(), **values)
    report.write_text(json.dumps(out, indent=2))
    print(json.dumps({'status': status, **values}), flush=True)

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()

started = time.monotonic()
child = None
try:
    if not archive.exists():
        emit('downloading')
        partial = archive.with_suffix('.zip.part')
        code = 'import gdown,sys; r=gdown.download(id=sys.argv[1],output=sys.argv[2],quiet=True); sys.exit(0 if r else 2)'
        child = subprocess.Popen([sys.executable, '-c', code, ident, str(partial)],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        while child.poll() is None:
            if time.monotonic() - started > 600: raise RuntimeError('metadata_download_timeout')
            if os.statvfs(root).f_bavail * os.statvfs(root).f_frsize < 16 * 2**30: raise RuntimeError('disk_reserve')
            # gdown may use a randomized intermediate filename before rename.
            total = sum(p.stat().st_size for p in dest.iterdir() if p.is_file())
            if total > out['archive_limit_bytes']: raise RuntimeError('annotation_archive_size_cap')
            time.sleep(1)
        assert child.returncode == 0, 'public_annotation_download_failed'
        assert partial.is_file() and 0 < partial.stat().st_size <= out['archive_limit_bytes']
        partial.replace(archive)
    emit('inventory')
    with zipfile.ZipFile(archive) as zipped:
        members = [{'name': entry.filename, 'bytes': entry.file_size,
                    'compressed_bytes': entry.compress_size, 'crc32': format(entry.CRC, '08x')}
                   for entry in zipped.infolist() if not entry.is_dir()]
        assert any(Path(entry['name']).name == 'rel.json' for entry in members), 'Expected RelTR VG relation metadata'
        assert all(Path(entry['name']).suffix.lower() != '.jpg' for entry in members), 'Unexpected image archive'
        assert sum(entry['bytes'] for entry in members) <= 2 * 2**30, 'Metadata uncompressed size cap'
    emit('download_complete_parity_pending', archive_bytes=archive.stat().st_size,
         archive_sha256=digest(archive), members=members)
except Exception as error:
    if child is not None and child.poll() is None:
        child.terminate()
        try: child.wait(timeout=10)
        except subprocess.TimeoutExpired: child.kill(); child.wait(timeout=10)
    emit('failed', error_type=type(error).__name__, reason=str(error)[:200])
    raise
