"""Bounded official Flickr30k source-link audit; no image/model/test scoring."""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter, defaultdict
from html.parser import HTMLParser
import hashlib
import json
import re
import signal
import urllib.request

root = Path.home() / 'rsg-former'
shared = Path.home() / 'shared-assets/datasets/flickr30k_karpathy'
signal.alarm(120)
metadata = json.loads((shared / 'annotations/dataset_flickr30k.json').read_text())
test_names = {row['filename'] for row in metadata['images'] if row['split'] == 'test'}
assert len(test_names) == 1000
ids = {Path(name).stem: name for name in test_names}
urls = defaultdict(set)
tag_counts = Counter()
host_counts = Counter()

class Links(HTMLParser):
    def handle_starttag(self, tag, attributes):
        for key, value in attributes:
            if key not in ('src', 'href') or not value: continue
            if value.startswith(('http://', 'https://')):
                tag_counts[tag + ':' + key] += 1
                host_counts[urllib.request.urlparse(value).hostname if hasattr(urllib.request, 'urlparse') else value.split('/')[2]] += 1
                match = re.search(r'/([0-9]+)_[^/]+\.(?:jpg|jpeg)(?:\?|$)', value, re.I)
                if match and match[1] in ids:
                    urls[ids[match[1]]].add(value)

source = 'https://shannon.cs.illinois.edu/DenotationGraph/data/flickr30k.html'
with urllib.request.urlopen(source, timeout=25) as response:
    declared = int(response.headers.get('Content-Length', '0'))
    assert declared <= 18 * 2**20
    raw = response.read(18 * 2**20 + 1)
assert 0 < len(raw) <= 18 * 2**20
parser = Links()
parser.feed(raw.decode('utf8', errors='replace'))
folder = shared / 'evaluation_only/source_metadata'
folder.mkdir(parents=True, exist_ok=True)
# Official gallery contains evaluation descriptions. It never becomes prompt/dev input.
(folder / 'official_flickr30k_gallery.html').write_bytes(raw)
manifest = {name: sorted(values) for name, values in sorted(urls.items())}
target = folder / 'official_test_image_links.json'
target.write_text(json.dumps(manifest, indent=2))
box = 'https://uofi.box.com/s/1cpolrtkckn4hxr1zhmfg0ln9veo6jpl'
try:
    with urllib.request.urlopen(box, timeout=25) as response:
        box_state = {'http_status': response.status, 'final_url': response.url,
                     'content_type': response.headers.get('Content-Type'),
                     'bounded_bytes_read': len(response.read(65536))}
except Exception as error:
    box_state = {'error_type': type(error).__name__}
out = {'status': 'SOURCE_LINK_AUDIT_COMPLETE', 'utc': datetime.now(timezone.utc).isoformat(),
       'official_gallery_url': source, 'gallery_bytes': len(raw),
       'gallery_sha256': hashlib.sha256(raw).hexdigest(), 'links_tag_counts': dict(tag_counts),
       'links_host_counts': dict(host_counts), 'karpathy_test_images': len(test_names),
       'test_images_with_direct_jpeg_links': len(manifest),
       'images_with_multiple_direct_links': sum(len(values) > 1 for values in manifest.values()),
       'test_link_manifest_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
       'example_test_image_links': list(manifest.items())[:2], 'official_box_access': box_state,
       'images_downloaded': 0, 'test_scored': False, 'prompt_tuned': False,
       'reference_descriptions_used_for_selection': False,
       'remaining': ['Verify original official dataset image release/pixel provenance before caption scoring',
                     'Download only missing TEST and independent development identities within storage caps from authorized sources']}
text = json.dumps(out, indent=2)
(root / 'reports/FLICKR_OFFICIAL_SOURCE_LINK_AUDIT.json').write_text(text)
print(text)
