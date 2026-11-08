"""Prepare checksum-verified project-local JRE, without shared installation or PATH edits."""
from pathlib import Path, PurePosixPath
from datetime import datetime, timezone
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import tarfile
import urllib.request
import urllib.error
from urllib.parse import urlparse

signal.alarm(180)
root = Path.home() / 'rsg-former'
tools = root / '.tools/java'
tools.mkdir(parents=True, exist_ok=True)
api = 'https://api.adoptium.net/v3/assets/latest/8/hotspot?architecture=x64&image_type=jre&os=linux&vendor=eclipse'
metadata_limit = 4 * 1024 * 1024
adoptium_api_status = 200
try:
    with urllib.request.urlopen(api, timeout=20) as response:
        raw = response.read(metadata_limit + 1)
    assert len(raw) <= metadata_limit
    assets = json.loads(raw)
    assert len(assets) == 1
    asset = assets[0]
except urllib.error.HTTPError as error:
    if error.code != 403:
        raise
    adoptium_api_status = 403
    # Public releases in Adoptium's own GitHub repository are an official alternate source.
    api = 'https://api.github.com/repos/adoptium/temurin8-binaries/releases/latest'
    with urllib.request.urlopen(api, timeout=20) as response:
        raw = response.read(metadata_limit + 1)
    assert len(raw) <= metadata_limit
    release = json.loads(raw)
    candidates = [item for item in release['assets']
                  if re.fullmatch(r'OpenJDK8U-jre_x64_linux_hotspot_[A-Za-z0-9_.-]+\.tar\.gz', item['name'])]
    assert len(candidates) == 1
    binary = candidates[0]
    checksums = [item for item in release['assets'] if item['name'] == binary['name'] + '.sha256.txt']
    assert len(checksums) == 1
    checksum_url = checksums[0]['browser_download_url']
    assert checksum_url.startswith('https://github.com/adoptium/temurin8-binaries/releases/download/')
    with urllib.request.urlopen(checksum_url, timeout=20) as response:
        checksum_text = response.read(1025).decode('ascii')
    assert len(checksum_text) <= 1024
    hashes = re.findall(r'\b[0-9a-f]{64}\b', checksum_text)
    assert len(hashes) == 1
    asset = {'release_name': release['tag_name'], 'version': {'major': 8},
             'binary': {'architecture': 'x64', 'os': 'linux', 'image_type': 'jre', 'jvm_impl': 'hotspot',
                        'package': {'name': binary['name'], 'size': binary['size'],
                                    'checksum': hashes[0], 'link': binary['browser_download_url']}}}
package = asset['binary']['package']
assert asset['binary']['architecture'] == 'x64' and asset['binary']['os'] == 'linux'
assert asset['binary']['image_type'] == 'jre' and asset['binary']['jvm_impl'] == 'hotspot'
assert asset['version']['major'] == 8
name, size, checksum, url = package['name'], package['size'], package['checksum'], package['link']
assert re.fullmatch(r'OpenJDK8U-jre_x64_linux_hotspot_[A-Za-z0-9_.-]+\.tar\.gz', name)
assert re.fullmatch(r'[0-9a-f]{64}', checksum)
assert 0 < size < 64 * 1024 ** 2
parsed = urlparse(url)
assert parsed.scheme == 'https' and parsed.netloc == 'github.com'
assert parsed.path.startswith('/adoptium/temurin8-binaries/releases/download/')
free_before = shutil.disk_usage(root).free
assert free_before >= 16 * 1024 ** 3 + 320 * 1024 ** 2, 'disk_reserve_gate'
archive = tools / name
downloaded = 0
if archive.exists():
    assert archive.stat().st_size == size
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == checksum
else:
    partial = archive.with_name(archive.name + '.part')
    assert not partial.exists(), 'preserve_partial_for_audit'
    digest = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=20) as response, partial.open('xb') as target:
        declared = response.headers.get('Content-Length')
        if declared is not None:
            assert int(declared) == size
        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            downloaded += len(block)
            assert downloaded <= size
            digest.update(block)
            target.write(block)
    assert downloaded == size and digest.hexdigest() == checksum
    partial.rename(archive)

destination = tools / ('temurin8-' + checksum[:12])
destination_absolute = destination.resolve()
with tarfile.open(archive, 'r:gz') as bundle:
    members = bundle.getmembers()
    assert len(members) < 2000
    unpacked = sum(member.size for member in members if member.isfile())
    assert unpacked < 256 * 1024 ** 2
    roots = set()
    for member in members:
        path = PurePosixPath(member.name)
        assert not path.is_absolute() and '..' not in path.parts
        assert path.parts
        roots.add(path.parts[0])
        target = (destination / member.name).resolve()
        assert target == destination_absolute or destination_absolute in target.parents
        assert member.isfile() or member.isdir() or member.issym() or member.islnk()
        if member.issym() or member.islnk():
            link = PurePosixPath(member.linkname)
            assert not link.is_absolute()
            resolved_link = ((target.parent if member.issym() else destination) / member.linkname).resolve()
            assert resolved_link == destination_absolute or destination_absolute in resolved_link.parents
        member.mode &= 0o777
    assert len(roots) == 1
    java = destination / next(iter(roots)) / 'bin/java'
    if not destination.exists():
        destination.mkdir()
        # Every archive member and link has been validated against this dedicated root.
        bundle.extractall(destination, members=members)
    assert java.is_file() and os.access(java, os.X_OK)
completed = subprocess.run([str(java), '-Xmx64m', '-XX:ActiveProcessorCount=1', '-version'],
                           capture_output=True, text=True, timeout=10)
assert completed.returncode == 0
report = {
    'status': 'PROJECT_LOCAL_JRE_PREPARED', 'utc': datetime.now(timezone.utc).isoformat(),
    'source_api': api, 'adoptium_api_http_status': adoptium_api_status,
    'source_documentation': 'https://adoptium.net/installation/archives',
    'metadata_sha256': hashlib.sha256(raw).hexdigest(), 'release_name': asset['release_name'],
    'archive_url': url, 'archive_sha256': checksum, 'archive_bytes': size,
    'network_archive_bytes_downloaded': downloaded, 'unpacked_regular_file_bytes': unpacked,
    'java_path_alias': str(java).replace(str(Path.home()), '${REMOTE_HOME}'),
    'java_version': completed.stderr.splitlines(),
    'java_executable_sha256': hashlib.sha256(java.read_bytes()).hexdigest(),
    'disk_free_before': free_before, 'disk_free_after': shutil.disk_usage(root).free,
    'shared_environment_modified': False, 'global_path_modified': False,
    'gpu_used': False, 'caption_benchmark_scored': False,
    'remaining': ['Bounded actual PTBTokenizer/METEOR/SPICE synthetic fixtures with child-only Java PATH and memory/thread caps'],
}
(root / 'reports/CAPTION_JAVA_PREPARATION.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report))
