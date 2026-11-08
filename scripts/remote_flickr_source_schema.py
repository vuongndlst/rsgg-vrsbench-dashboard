"""Inspect URL/tag/Box file metadata only; never print gallery descriptions."""
from pathlib import Path
from html.parser import HTMLParser
from collections import Counter
import json
import re
import signal
import urllib.request
from urllib.parse import urlparse, urljoin
signal.alarm(90)
root = Path.home() / 'rsg-former'
source = Path.home() / 'shared-assets/datasets/flickr30k_karpathy/evaluation_only/source_metadata'
raw = (source / 'official_flickr30k_gallery.html').read_text()
tags = Counter()
attributes = Counter()
url_values = []
class Schema(HTMLParser):
    def handle_starttag(self, tag, attrs):
        tags[tag] += 1
        for key, value in attrs:
            attributes[tag + ':' + key] += 1
            if key in ('href', 'src') and value:
                url_values.append(urljoin('https://shannon.cs.illinois.edu/DenotationGraph/data/flickr30k.html', value))
Schema().feed(raw)
plain_urls = re.findall(r'https?://[^\s<>"\']+', raw)
box_url = 'https://uofi.app.box.com/s/1cpolrtkckn4hxr1zhmfg0ln9veo6jpl'
with urllib.request.urlopen(box_url, timeout=25) as response:
    box = response.read(1024 * 1024).decode('utf8', errors='replace')
(source / 'official_box_public_page.html').write_text(box)
names = sorted(set(re.findall(r'[^"<>\s:/]{1,120}\.(?:tar\.gz|zip|tgz)', box)))
out = {'gallery_tag_counts': dict(tags), 'gallery_attribute_counts': dict(attributes),
       'gallery_url_hosts': dict(Counter(urlparse(url).hostname for url in url_values + plain_urls)),
       'gallery_image_url_examples': [url for url in url_values + plain_urls if '.jpg' in url][:3],
       'box_archive_names': names, 'box_file_paths': sorted(set(re.findall(r'/file/\d+', box))),
       'box_script_ids': re.findall(r'<script[^>]+id="([^"]+)"', box),
       'box_contains_download_url_key': bool(re.search('downloadUrl|download_url', box)),
       'box_json_key_counts': Counter(re.findall(r'"(\w+)"\s*:', box)).most_common(30),
       'captions_printed': False, 'images_downloaded': 0}
print(json.dumps(out, indent=2))
