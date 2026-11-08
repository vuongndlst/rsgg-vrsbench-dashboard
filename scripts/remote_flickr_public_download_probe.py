"""Bounded public shared-file download check, without auth or archive storage."""
from pathlib import Path
from datetime import datetime, timezone
import json
import signal
import urllib.request
from urllib.parse import urlencode, urlparse
signal.alarm(60)
root = Path.home() / 'rsg-former'
metadata = json.loads((root / 'reports/FLICKR_PUBLIC_DOWNLOAD_METADATA.json').read_text())
assert metadata['public_box_flags']['canDownload'] == ['true']
archive = next(row for row in metadata['public_archive_metadata'] if row['size'] is not None)
assert archive['name'] == 'flickr30k-images.tar.gz' and archive['size'] == 4375349371
# Box's normal public shared-file download route. No private API/token/login.
url = 'https://uofi.app.box.com/index.php?' + urlencode({
    'rm': 'box_download_shared_file', 'shared_name': '1cpolrtkckn4hxr1zhmfg0ln9veo6jpl',
    'file_id': archive['id']})
out = {'utc': datetime.now(timezone.utc).isoformat(), 'public_file_id': archive['id'],
       'official_archive_bytes': archive['size'], 'archive_stored': False, 'auth_used': False,
       'test_scored': False, 'bytes_read_cap': 1024}
try:
    request = urllib.request.Request(url, headers={'Range': 'bytes=0-1023'})
    with urllib.request.urlopen(request, timeout=25) as response:
        raw = response.read(1024)
        out.update(http_status=response.status, final_host=urlparse(response.url).hostname,
                   content_type=response.headers.get('Content-Type'),
                   content_length=response.headers.get('Content-Length'),
                   content_range=response.headers.get('Content-Range'), bytes_read=len(raw),
                   gzip_magic_matches=raw.startswith(b'\x1f\x8b'))
except Exception as error:
    out.update(error_type=type(error).__name__, http_status=getattr(error, 'code', None),
               bytes_read=0, gzip_magic_matches=False)
out['download_source_verified'] = out.get('gzip_magic_matches', False)
text = json.dumps(out, indent=2)
(root / 'reports/FLICKR_PUBLIC_DOWNLOAD_PROBE.json').write_text(text)
print(text)
