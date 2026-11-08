"""Bounded official COCO metadata-only ZIP ranges; held-out identities, no scoring."""
from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import urlparse
import hashlib
import io
import json
import os
import re
import shutil
import signal
import urllib.request
import zipfile

signal.alarm(180)
os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
root = Path.home() / 'rsg-former'
folder = Path.home() / 'shared-assets/datasets/coco_karpathy/evaluation_only/source_metadata'
folder.mkdir(parents=True, exist_ok=True)
assert shutil.disk_usage(root).free > 16 * 1024 ** 3 + 128 * 1024 ** 2
page_url = 'https://raw.githubusercontent.com/cocodataset/cocodataset.github.io/master/dataset/download.htm'
official_link = 'http://images.cocodataset.org/annotations/annotations_trainval2014.zip'
# Use the archive endpoint linked directly by the official download source.
# A prior public GCP bucket attempt returned 404; preserve that failure separately.
url = official_link
report_path = root / 'reports/COCO_PROTECTED_IDENTITY_PREPARATION.json'
report = {'utc': datetime.now(timezone.utc).isoformat(), 'status': 'preparing',
          'official_source_page': page_url, 'official_archive_link': official_link,
          'archive_url': url, 'image_pixels_downloaded': False, 'benchmark_test_scored': False,
          'live_v2_modified': False, 'training_manifest_modified': False,
          'reference_text_used_for_identity_matching': False}

class RangeZip(io.RawIOBase):
    def __init__(self):
        with urllib.request.urlopen(urllib.request.Request(url, method='HEAD'), timeout=20) as response:
            self.size = int(response.headers['Content-Length'])
            self.etag = response.headers.get('ETag')
        assert 0 < self.size < 300 * 1024 ** 2
        assert self.etag, 'stable_range_source_requires_ETag'
        self.position, self.total, self.records = 0, 0, []

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        assert 0 <= self.position <= self.size
        return self.position

    def read(self, count=-1):
        count = min(self.size - self.position, count if count >= 0 else self.size)
        if count == 0:
            return b''
        assert count < 20 * 1024 ** 2 and self.total + count <= 32 * 1024 ** 2, 'range_byte_budget'
        start, end = self.position, self.position + count - 1
        request = urllib.request.Request(url, headers={'Range': f'bytes={start}-{end}', 'If-Match': self.etag})
        with urllib.request.urlopen(request, timeout=20) as response:
            assert response.status == 206, 'server_must_honor_range_no_full_download'
            assert response.headers['Content-Range'] == f'bytes {start}-{end}/{self.size}'
            assert response.headers.get('ETag') == self.etag
            data = response.read(count + 1)
        assert len(data) == count
        self.position += count
        self.total += count
        self.records.append({'start': start, 'end': end, 'bytes': count,
                             'sha256': hashlib.sha256(data).hexdigest()})
        return data

try:
    with urllib.request.urlopen(page_url, timeout=20) as response:
        page = response.read(128 * 1024 + 1)
    assert len(page) <= 128 * 1024 and official_link.encode() in page
    assert b'gs://images.cocodataset.org/annotations' in page
    page_path = folder / 'official_coco_download_source.htm'
    if page_path.exists():
        assert page_path.read_bytes() == page, 'preserve_changed_official_source'
    else:
        page_path.write_bytes(page)
    source = RangeZip()
    member = 'annotations/captions_val2014.json'
    with zipfile.ZipFile(source) as bundle:
        info = bundle.getinfo(member)
        assert info.file_size < 64 * 1024 ** 2 and info.compress_size < 20 * 1024 ** 2
        raw = bundle.read(member)  # ZIP reader checks member CRC; no other member is read.
    target = folder / 'captions_val2014.json'
    if target.exists():
        assert target.read_bytes() == raw, 'preserve_existing_member'
    else:
        target.write_bytes(raw)
    images = json.loads(raw)['images']  # Only image identity fields are retained below.
    karpathy_path = Path.home() / 'shared-assets/datasets/coco_karpathy/annotations/dataset_coco.json'
    karpathy_raw = karpathy_path.read_bytes()
    assert hashlib.sha256(karpathy_raw).hexdigest() == '2fd999220673258012acfb411a4e7e66af7d488050b2519b0badcc49b7600b8d'
    protected = {int(item['cocoid']): item['split'] for item in json.loads(karpathy_raw)['images']
                 if item['split'] in ['val', 'test']}
    identities, covered = [], set()
    for item in images:
        coco_id = int(item['id'])
        if coco_id not in protected:
            continue
        assert coco_id not in covered
        covered.add(coco_id)
        flickr_url = item.get('flickr_url') or ''
        parsed = urlparse(flickr_url)
        host = parsed.hostname or ''
        assert host == 'flickr.com' or host.endswith('.flickr.com') or host == 'staticflickr.com' or host.endswith('.staticflickr.com')
        flickr_id = Path(parsed.path).stem.split('_')[0]
        assert re.fullmatch(r'[0-9]+', flickr_id)
        identities.append({'coco_id': coco_id, 'split': protected[coco_id],
                           'flickr_id': flickr_id, 'flickr_url': flickr_url,
                           'filename': item['file_name'], 'width': item['width'], 'height': item['height']})
    assert covered == set(protected), 'full_protected_coco_identity_coverage_required'
    metadata_raw = (root / 'data/vg150/image_data.json').read_bytes()
    metadata = json.loads(metadata_raw)
    flickr = {item['flickr_id']: item for item in identities}
    assert len(flickr) == len(identities)
    previous_path = root / 'artifacts/caption_readiness/canonical_internal_external_identity_splits_v2.json'
    previous_raw = previous_path.read_bytes()
    assert hashlib.sha256(previous_raw).hexdigest() == 'ade090fd4975e12e1ade2ac16a734f7b167f6630b38d1710cefa4176523f7b16'
    previous = json.loads(previous_raw)
    roles = {int(identity): split for split, ids in previous['split_image_ids'].items() for identity in ids}
    matches = []
    for item in metadata:
        vg_id = int(item['image_id'])
        if vg_id not in roles:
            continue
        match = flickr.get(str(item.get('flickr_id')))
        if match:
            matches.append({'vg_image_id': vg_id, 'vg_split': roles[vg_id],
                            'coco_id': match['coco_id'], 'external_split': match['split'],
                            'evidence': 'official_COCO_val2014_images_flickr_url'})
    artifact = root / 'artifacts/caption_readiness/coco_protected_flickr_identities.json'
    artifact_raw = json.dumps({'identities': sorted(identities, key=lambda item: item['coco_id']),
                               'new_matches_after_previous_exclusions': matches}, separators=(',', ':')).encode()
    if artifact.exists():
        assert artifact.read_bytes() == artifact_raw
    else:
        artifact.write_bytes(artifact_raw)
    report.update(status='FULL_PROTECTED_COCO_METADATA_IDENTITIES_PREPARED',
                  source_page_sha256=hashlib.sha256(page).hexdigest(),
                  archive_bytes=source.size, archive_etag=source.etag,
                  full_archive_downloaded=False, full_archive_sha256=None,
                  metadata_range_download_bytes=source.total, range_records=source.records,
                  member=member, member_bytes=len(raw), member_crc32=f'{info.CRC:08x}',
                  member_sha256=hashlib.sha256(raw).hexdigest(), official_val2014_images=len(images),
                  protected_coco_ids=len(protected), protected_ids_with_flickr_identity=len(identities),
                  new_matches_after_previous_exclusions=len(matches),
                  new_match_counts={split: len({match['vg_image_id'] for match in matches if match['vg_split'] == split})
                                    for split in ['train', 'val']},
                  identity_artifact_sha256=hashlib.sha256(artifact_raw).hexdigest(),
                  previous_manifest_sha256=hashlib.sha256(previous_raw).hexdigest(),
                  vg_metadata_sha256=hashlib.sha256(metadata_raw).hexdigest(),
                  source_contains_caption_text=True, only_image_identifier_fields_used=True,
                  clean_release_certified=False,
                  remaining=['Version additional verified identity exclusions if any',
                             'Full protected exact/perceptual image audit before clean-release training'])
except Exception as error:
    report.update(status='FAILED_BEFORE_IDENTITY_RELEASE', error_type=type(error).__name__, error=str(error))
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report))
    raise
report_path.write_text(json.dumps(report, indent=2))
print(json.dumps(report))
