"""Test-only, one-image-at-a-time CVDF reader without a persistent pixel cache.

No reference text, labels, GT graphs, model, resizing, or score-based selection.
Raw CVDF orientation is preserved; metadata Rotation is recorded, not applied.
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import io
import json
import re
import urllib.request

from PIL import Image


@dataclass
class TestImage:
    image: Image.Image
    provenance: dict


class OpenImagesTestReader:
    def __init__(self, shared_root, *, manifest_sha256, allow_network=False,
                 single_image_bytes_cap=8 * 2**20, network_bytes_cap=0,
                 known_pixel_records=()):
        self.root = Path(shared_root)
        raw = (self.root / 'test_identity_manifest.json').read_bytes()
        if hashlib.sha256(raw).hexdigest() != manifest_sha256:
            raise ValueError('test_identity_manifest_hash_mismatch')
        manifest = json.loads(raw)
        self.records = {row['ImageID']: row for row in manifest['images']}
        self.ids = tuple(sorted(manifest['caption_test_ids']))
        if len(self.ids) != len(set(self.ids)) or set(self.ids) != set(self.records):
            raise ValueError('test_identity_membership_mismatch')
        if any(not re.fullmatch('[0-9a-f]{16}', identity) for identity in self.ids):
            raise ValueError('invalid_test_identity')
        self.expected_pixels = {row['image_id']: row for row in known_pixel_records}
        if not set(self.expected_pixels).issubset(self.records):
            raise ValueError('pixel_manifest_contains_non_test_identity')
        self.allow_network = allow_network
        self.single_cap = int(single_image_bytes_cap)
        self.network_cap = int(network_bytes_cap)
        assert self.single_cap > 0 and self.network_cap >= 0
        self.downloaded_bytes = 0

    def __len__(self):
        return len(self.ids)

    def read(self, identity):
        if identity not in self.records:
            raise ValueError('identity_not_in_locked_test_manifest')
        path = self.root / 'images' / (identity + '.jpg')
        source = 'https://open-images-dataset.s3.amazonaws.com/test/' + identity + '.jpg'
        if path.is_file():
            if path.stat().st_size > self.single_cap:
                raise ValueError('single_image_byte_cap_exceeded')
            raw = path.read_bytes()
            loaded_from = 'existing_shared_test_image'
        else:
            if not self.allow_network:
                raise RuntimeError('network_disabled_missing_test_image')
            remaining = self.network_cap - self.downloaded_bytes
            if remaining <= 0:
                raise RuntimeError('network_byte_budget_exhausted')
            with urllib.request.urlopen(source, timeout=20) as response:
                size = int(response.headers.get('Content-Length', '0'))
                if not 0 < size <= min(self.single_cap, remaining):
                    raise RuntimeError('download_size_unknown_or_byte_cap_exceeded')
                if 'image' not in response.headers.get('Content-Type', ''):
                    raise ValueError('unexpected_test_pixel_content_type')
                raw = response.read(size)
                self.downloaded_bytes += len(raw)
                if len(raw) != size:
                    raise ValueError('incomplete_test_pixel_download')
            loaded_from = 'in_memory_public_cvdf_download'
        if not raw:
            raise ValueError('empty_test_pixel')
        digest = hashlib.sha256(raw).hexdigest()
        known = self.expected_pixels.get(identity)
        if known and (digest != known['sha256'] or len(raw) != known['bytes']):
            raise ValueError('known_test_pixel_hash_mismatch')
        with Image.open(io.BytesIO(raw)) as encoded:
            if encoded.format != 'JPEG':
                raise ValueError('unexpected_test_pixel_format')
            # CVDF provides longest side <=1024; unexpected sources fail visibly.
            if max(encoded.size) > 1024:
                raise ValueError('cvdf_image_dimension_contract_mismatch')
            encoded.verify()
        with Image.open(io.BytesIO(raw)) as decoded:
            decoded.load()
            size = decoded.size
            exif_orientation = decoded.getexif().get(274)
            image = decoded.convert('RGB')
        provenance = {'image_id': identity, 'source': source, 'loaded_from': loaded_from,
                      'bytes': len(raw), 'sha256': digest, 'dimensions': list(size),
                      'decoder': 'Pillow JPEG full decode -> RGB',
                      'orientation': 'raw CVDF pixel matrix; no EXIF transpose or Rotation applied',
                      'metadata_rotation': self.records[identity]['Rotation'],
                      'exif_orientation': exif_orientation, 'pixel_cache_written': False}
        return TestImage(image, provenance)
