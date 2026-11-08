"""Bounded official VG region metadata repair, isolated from live SGDet inputs."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import time
import urllib.request
import zipfile

root = Path.home() / 'rsg-former'
dest = root / 'data/vg_caption'
dest.mkdir(parents=True, exist_ok=True)
report = root / 'reports/VG_REGION_REFERENCE_PREPARATION.json'
url = 'https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/region_descriptions.json.zip'
archive = dest / 'region_descriptions.json.zip'
target = dest / 'region_descriptions.json'
started = time.monotonic()
state = {'source_url': url, 'old_asset_preserved': True, 'image_archives_extracted': False,
         'purpose': 'human evaluation references only; never graph-to-text prompt input'}

def emit(status, **extra):
    state.update(status=status, updated_utc=datetime.now(timezone.utc).isoformat(), **extra)
    report.write_text(json.dumps(state, indent=2))
    print(json.dumps({'status': status, **extra}), flush=True)

def guard():
    assert time.monotonic() - started < 600, 'Bounded metadata preparation timeout'
    assert os.statvfs(root).f_bavail * os.statvfs(root).f_frsize > 16 * 2**30, 'Disk reserve'

def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()

def records(path):
    # Top-level array streaming using stdlib: bounded memory per image record.
    decoder = json.JSONDecoder()
    with path.open(encoding='utf8') as stream:
        buffer = ''; eof = False; opened = False; closed = False; expect_value = True; first = True
        while True:
            guard()
            if not eof:
                chunk = stream.read(1024 * 1024)
                buffer += chunk
                eof = not chunk
            cursor = 0
            while True:
                while cursor < len(buffer) and buffer[cursor].isspace(): cursor += 1
                if not opened:
                    if cursor == len(buffer): break
                    assert buffer[cursor] == '['
                    opened = True; cursor += 1; continue
                if closed:
                    assert not buffer[cursor:].strip(), 'Trailing JSON content'
                    cursor = len(buffer); break
                if cursor == len(buffer): break
                if buffer[cursor] == ']':
                    assert first or not expect_value, 'Trailing array comma'
                    closed = True; cursor += 1; continue
                if not expect_value:
                    assert buffer[cursor] == ',', 'Expected array separator'
                    cursor += 1; expect_value = True; continue
                try: item, end = decoder.raw_decode(buffer, cursor)
                except json.JSONDecodeError:
                    if eof: raise
                    break
                assert isinstance(item, dict)
                yield item
                first = False; expect_value = False; cursor = end
            buffer = buffer[cursor:]
            assert len(buffer) < 16 * 1024 * 1024, 'Unexpectedly large image annotation record'
            if eof:
                assert closed and not buffer.strip(), 'Incomplete top-level array'
                return

try:
    guard()
    previous = json.loads(report.read_text()) if report.exists() else {}
    if previous.get('status') == 'complete' and target.is_file() and sha(target) == previous['json_sha256']:
        print(json.dumps({'status': 'already_complete'})); raise SystemExit
    emit('downloading')
    if not archive.exists():
        part = archive.with_suffix('.zip.part')
        with urllib.request.urlopen(url, timeout=30) as response, part.open('wb') as stream:
            size = int(response.headers.get('Content-Length', '0'))
            assert 0 < size <= 200 * 2**20, 'Unexpected archive size'
            downloaded = 0
            while True:
                guard(); chunk = response.read(1024 * 1024)
                if not chunk: break
                stream.write(chunk); downloaded += len(chunk)
                assert downloaded <= 200 * 2**20
            assert downloaded == size, 'Incomplete download'
        part.replace(archive)
    emit('validating_archive', archive_bytes=archive.stat().st_size, archive_sha256=sha(archive))
    with zipfile.ZipFile(archive) as zipped:
        entries = [entry for entry in zipped.infolist() if Path(entry.filename).name == 'region_descriptions.json']
        assert len(entries) == 1
        entry = entries[0]; assert 0 < entry.file_size <= 2 * 2**30
        part = target.with_suffix('.json.part')
        with zipped.open(entry) as source, part.open('wb') as stream:
            for block in iter(lambda: source.read(1024 * 1024), b''):
                guard(); stream.write(block)
        assert part.stat().st_size == entry.file_size
        # ZipFile verifies this member's CRC at EOF; no extraction of other files.
    emit('validating_json', json_bytes=part.stat().st_size)
    ids = set(); region_count = 0; phrases = 0; invalid_geometry = 0
    for item in records(part):
        identity = int(item.get('id', item.get('image_id')))
        assert identity not in ids; ids.add(identity)
        for region in item['regions']:
            region_count += 1
            phrases += bool(region.get('phrase', '').strip())
            invalid_geometry += int(region['width'] <= 0 or region['height'] <= 0 or region['x'] < 0 or region['y'] < 0)
    assert len(ids) > 100000 and region_count > 1000000, 'Incomplete reference coverage'
    part.replace(target)
    (dest / 'region_image_ids.json').write_text(json.dumps(sorted(ids), separators=(',', ':')))
    emit('complete', json_bytes=target.stat().st_size, json_sha256=sha(target),
         image_records=len(ids), region_count=region_count, nonempty_phrases=phrases,
         invalid_geometry_records=invalid_geometry,
         canonical_path='${REMOTE_HOME}/rsg-former/data/vg_caption/region_descriptions.json')
except Exception as error:
    emit('failed', error_type=type(error).__name__, reason=str(error)[:240]); raise
