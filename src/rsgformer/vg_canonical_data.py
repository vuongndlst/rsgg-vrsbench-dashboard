"""Independent canonical VG adapter for a future preregistered experiment.

Matches the pinned Salience-SGG COCO target preparation before transforms.
Does not change VG150/H5 or the live v2 experiment. Images stay in original ZIPs.
"""
from collections import defaultdict
from pathlib import Path
import hashlib
import io
import json
import zipfile
import numpy as np
from PIL import Image
from .data import ImageGraphSample


class VGCanonical:
    def __init__(self, annotation_archive, image_root, split='train', exclude_image_ids=()):
        assert split in ('train', 'val', 'test')
        self.archive = Path(annotation_archive)
        self.image_root = Path(image_root)
        self.split = split
        with zipfile.ZipFile(self.archive) as zipped:
            raw = zipped.read('vg/' + split + '.json')
            self.annotation_sha256 = hashlib.sha256(raw).hexdigest()
            data = json.loads(raw)
            raw = zipped.read('vg/rel.json')
            self.relation_sha256 = hashlib.sha256(raw).hexdigest()
            relation_data = json.loads(raw)
        self.object_names = {int(c['id']): c['name'] for c in data['categories'] if int(c['id']) > 0}
        self.predicate_names = dict(enumerate(relation_data['rel_categories']))
        assert set(self.object_names) == set(range(1, 151))
        assert len(self.predicate_names) == 51
        self.image_info = {int(image['id']): image for image in data['images']}
        assert len(self.image_info) == len(data['images'])
        self.source_ordered_ids = [int(image['id']) for image in data['images']]
        # torchvision.datasets.CocoDetection sorts self.coco.imgs.keys().
        protected = set(map(int, exclude_image_ids))
        self.ids = sorted(set(self.image_info) - protected)
        self.excluded_present_ids = sorted(set(self.image_info) & protected)
        self.annotations = defaultdict(list)
        for annotation in data['annotations']:
            self.annotations[int(annotation['image_id'])].append(annotation)
        self.relations = relation_data[split]
        self._zips = None

    def __len__(self):
        return len(self.ids)

    def split_image_ids(self):
        return list(self.ids)

    def split_hash(self):
        return hashlib.sha256(json.dumps(self.ids, separators=(',', ':')).encode()).hexdigest()

    def __getitem__(self, index):
        identity = self.ids[index]
        image = self.image_info[identity]
        width, height = int(image['width']), int(image['height'])
        annotations = [a for a in self.annotations[identity] if not a.get('iscrowd', 0)]
        boxes = np.array([a['bbox'] for a in annotations], dtype=np.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, width)
        boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, height)
        keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        labels = np.array([a['category_id'] for a in annotations], dtype=np.int64)
        global_ids = np.array([a['id'] for a in annotations], dtype=np.int64)
        relations = np.array(self.relations[str(identity)], dtype=np.int64).reshape(-1, 3)
        # Upstream graph construction assumes object order survives preparation.
        # Refuse an unsupported remapping rather than silently changing the GT.
        if len(annotations) != len(self.annotations[identity]) or not keep.all():
            raise ValueError('canonical_object_filter_requires_separate_native_graph_order_audit:' + str(identity))
        assert len(relations) == len(set(map(tuple, relations)))
        assert np.all((labels >= 1) & (labels <= 150))
        assert np.all((relations[:, 2] >= 1) & (relations[:, 2] <= 50))
        assert np.all((relations[:, :2] >= 0) & (relations[:, :2] < len(boxes)))
        return ImageGraphSample(identity, width, height, boxes, labels, relations, global_ids,
                                dataset='VG150 canonical RelTR',
                                coordinate_convention='native COCO-prepared float32 xyxy; x2=x+w, clip to image width/height')

    def native_target(self, index, queries=200):
        import torch
        sample = self[index]
        if len(sample.labels) > queries:
            raise ValueError('object_query_capacity_exceeded_requires_registered_exclusion')
        boxes = torch.from_numpy(sample.boxes.copy())
        boxes /= torch.tensor([sample.width, sample.height, sample.width, sample.height])
        boxes = torch.cat([(boxes[:, :2] + boxes[:, 2:]) / 2, boxes[:, 2:] - boxes[:, :2]], dim=1)
        rel = torch.zeros(queries, queries, 50)
        indices = torch.from_numpy(sample.relations)
        rel[indices[:, 0], indices[:, 1], indices[:, 2] - 1] = 1
        return {'labels': torch.from_numpy(sample.labels - 1), 'boxes': boxes,
                'rel': rel, 'image_id': torch.tensor([sample.image_id])}

    def _open_archives(self):
        self._zips = [zipfile.ZipFile(self.image_root / name) for name in ('images.zip', 'images2.zip')]
        self._member_index = {}
        for zi, zipped in enumerate(self._zips):
            for member in zipped.infolist():
                if not member.filename.lower().endswith('.jpg'): continue
                try: identity = int(Path(member.filename).stem)
                except ValueError: continue
                assert identity not in self._member_index
                self._member_index[identity] = (zi, member.filename)

    def image(self, index):
        if self._zips is None: self._open_archives()
        sample = self[index]
        zi, name = self._member_index[sample.image_id]
        with self._zips[zi].open(name) as stream:
            image = Image.open(io.BytesIO(stream.read())).convert('RGB')
        assert image.size == (sample.width, sample.height)
        return image

    def close(self):
        if self._zips:
            for zipped in self._zips: zipped.close()
        self._zips = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_zips'] = None
        state.pop('_member_index', None)
        return state
