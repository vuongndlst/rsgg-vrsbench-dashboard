import sys, time, io, zipfile, json
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms.functional as F

os_env = '1'
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
DEVICE = torch.device('cuda:0')

SHARED = Path.home() / 'shared-assets/datasets/vrsbench'
img_zip = SHARED / 'Images_train.zip'
ann_zip = SHARED / 'Annotations_train.zip'

CLASSES = [
    "background",
    "vehicle", "airplane", "ship", "harbor", "bridge", "tennis-court",
    "storage-tank", "baseball-diamond", "swimming-pool", "ground-track-field",
    "basketball-court", "roundabout", "windmill", "overpass", "soccer-ball-field",
    "expressway-service-area", "airport", "trainstation", "expressway-toll-station",
    "chimney", "stadium", "dam", "golffield", "helicopter", "container-crane", "helipad"
]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

class VRSBenchDetectionDataset(Dataset):
    def __init__(self, img_zip_path, ann_zip_path):
        self.img_zip_path = str(img_zip_path)
        self.ann_zip_path = str(ann_zip_path)
        self.img_zip = None
        self.ann_zip = None
        
        with zipfile.ZipFile(ann_zip_path, 'r') as z_ann, zipfile.ZipFile(img_zip_path, 'r') as z_img:
            img_names = {Path(f).name: f for f in z_img.namelist() if f.lower().endswith(('.png', '.jpg'))}
            self.samples = []
            for ann_file in z_ann.namelist():
                if not ann_file.endswith('.json') or ann_file.startswith('__MACOSX'):
                    continue
                base_stem = Path(ann_file).stem
                for ext in ['.png', '.jpg']:
                    cand = base_stem + ext
                    if cand in img_names:
                        self.samples.append({
                            'ann_path': ann_file,
                            'img_path': img_names[cand],
                            'img_id': base_stem
                        })
                        break

    def _open_zips(self):
        if self.img_zip is None:
            self.img_zip = zipfile.ZipFile(self.img_zip_path, 'r')
        if self.ann_zip is None:
            self.ann_zip = zipfile.ZipFile(self.ann_zip_path, 'r')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        self._open_zips()
        item = self.samples[idx]
        img_bytes = self.img_zip.read(item['img_path'])
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        w, h = img.size
        img_tensor = F.to_tensor(img)
        
        ann_data = json.loads(self.ann_zip.read(item['ann_path']).decode('utf-8'))
        objs = ann_data.get('objects', [])
        
        boxes = []
        labels = []
        for o in objs:
            cls_name = o.get('obj_cls', '')
            if cls_name not in CLASS_TO_IDX or cls_name == 'background':
                continue
            coord = o.get('obj_coord')
            if coord and len(coord) == 4:
                xmin, ymin, xmax, ymax = coord
                x1, y1, x2, y2 = float(xmin) * w, float(ymin) * h, float(xmax) * w, float(ymax) * h
            else:
                corner = o.get('obj_corner')
                if corner and len(corner) == 8:
                    xs = [float(corner[i]) * w for i in range(0, 8, 2)]
                    ys = [float(corner[i]) * h for i in range(1, 8, 2)]
                    x1, x2 = min(xs), max(xs)
                    y1, y2 = min(ys), max(ys)
                else:
                    continue
            
            x1 = max(0.0, min(x1, float(w)))
            y1 = max(0.0, min(y1, float(h)))
            x2 = max(0.0, min(x2, float(w)))
            y2 = max(0.0, min(y2, float(h)))
            
            if (x2 - x1) >= 2.0 and (y2 - y1) >= 2.0:
                boxes.append([x1, y1, x2, y2])
                labels.append(CLASS_TO_IDX[cls_name])
                
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)
            
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx]),
            'area': area,
            'iscrowd': iscrowd
        }
        return img_tensor, target

def collate_fn(batch):
    return tuple(zip(*batch))

print("Loading dataset...")
ds = VRSBenchDetectionDataset(img_zip, ann_zip)
loader = DataLoader(ds, batch_size=8, shuffle=True, num_workers=2, collate_fn=collate_fn)

print("Building model...")
model = fasterrcnn_resnet50_fpn(weights=None)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=len(CLASSES))
model.to(DEVICE)
model.train()

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

print("Running 10 benchmark steps...")
t0 = time.time()
trained_steps = 0
for i, (images, targets) in enumerate(loader):
    # filter targets
    valid_targets = [t for t in targets if len(t['boxes']) > 0]
    if not valid_targets:
        continue
    valid_imgs = [img for img, t in zip(images, targets) if len(t['boxes']) > 0]
    images = [img.to(DEVICE) for img in valid_imgs]
    targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in valid_targets]
    
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
    optimizer.zero_grad()
    losses.backward()
    optimizer.step()
    
    trained_steps += 1
    if trained_steps % 2 == 0:
        print(f"Step {trained_steps}: loss = {losses.item():.4f}")
    if trained_steps >= 10:
        break

t1 = time.time()
sec_per_step = (t1 - t0) / trained_steps
print(f"Time for {trained_steps} steps: {t1-t0:.2f}s ({sec_per_step:.3f} s/step)")
total_steps_epoch = len(loader)
print(f"Estimated time per epoch ({total_steps_epoch} steps): {total_steps_epoch * sec_per_step / 60.0:.1f} minutes")
print(f"Estimated time for 5 epochs: {5 * total_steps_epoch * sec_per_step / 60.0:.1f} minutes")

