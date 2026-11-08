"""Read public Box download metadata and check gallery image accessibility."""
from pathlib import Path
from datetime import datetime, timezone
import json
import re
import signal
import urllib.request
from urllib.parse import urlparse, urlunparse
from urllib.parse import urljoin
from html import unescape
signal.alarm(90)
root = Path.home() / 'rsg-former'
folder = Path.home() / 'shared-assets/datasets/flickr30k_karpathy/evaluation_only/source_metadata'
box = (folder / 'official_box_public_page.html').read_text()
download_values = []
source_candidates = []
for match in re.finditer(r'''(?:["']?downloadUrl["']?)\s*[:=]\s*(["'])(.*?)\1''', box):
    value = unescape(match[2].replace('\\/', '/'))
    value = urljoin('https://uofi.app.box.com', value)
    if isinstance(value, str) and value.startswith('https://'):
        parsed = urlparse(value)
        download_values.append({'host': parsed.hostname, 'query_present': bool(parsed.query),
                                'public_download_endpoint': 'box_download_shared_file' in value})
        source_candidates.append(value)
    else: download_values.append({'value_type': type(value).__name__})
if source_candidates:
    private_folder = root / '.tools/flickr_source'
    private_folder.mkdir(parents=True, exist_ok=True)
    (private_folder / 'public_download_candidates.json').write_text(json.dumps(source_candidates))
flags = {key: re.findall(r'"' + key + r'"\s*:\s*(true|false)', box)
         for key in ['canDownload', 'isDownloadable', 'isDownloadEnabled', 'isPasswordEnabled']}
archive_metadata = []
# Only return the archive's public metadata; avoid names/emails/tokens elsewhere on the page.
decoder = json.JSONDecoder()
for match in re.finditer(r'\{\s*"', box):
    try: value, _ = decoder.raw_decode(box[match.start():])
    except ValueError: continue
    if isinstance(value, dict) and value.get('name') == 'flickr30k-images.tar.gz':
        archive_metadata.append({key: value.get(key) for key in ('id', 'name', 'type', 'size', 'extension', 'isDownloadable')})
url = 'https://shannon.cs.illinois.edu/DenotationGraph/data/1000092795.jpg'
try:
    with urllib.request.urlopen(urllib.request.Request(url, method='HEAD'), timeout=20) as response:
        gallery = {'http_status': response.status, 'content_type': response.headers.get('Content-Type'),
                   'content_length': response.headers.get('Content-Length')}
except Exception as error:
    gallery = {'error_type': type(error).__name__, 'http_status': getattr(error, 'code', None)}
out = {'utc': datetime.now(timezone.utc).isoformat(), 'public_box_download_metadata': download_values,
       'public_box_flags': flags, 'public_archive_metadata': archive_metadata,
       'gallery_relative_jpeg_accessibility': gallery, 'signed_query_or_token_printed': False,
       'images_downloaded': 0, 'test_scored': False}
text = json.dumps(out, indent=2)
(root / 'reports/FLICKR_PUBLIC_DOWNLOAD_METADATA.json').write_text(text)
print(text)
