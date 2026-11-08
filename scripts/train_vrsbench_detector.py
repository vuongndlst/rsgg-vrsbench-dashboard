"""Full-scale Object Detector Training on VRSBench (20,264 images, 26 remote sensing classes).
Architecture: Faster R-CNN with ResNet-50-FPN Backbone.
Features:
- Fixed strictly to GPU 1 (CUDA_VISIBLE_DEVICES=1)
- Reads on-the-fly from Images_train.zip & Annotations_train.zip (Zero disk extraction footprint)
- 26 Geospatial categories + 1 background = 27 classes
- Correctly parses 'obj_coord' and 'obj_corner' scaled to [W, H]
- Mixed precision training (bfloat16)
- Logs step losses and epoch metrics to JSON for reporting & visualization
- Saves checkpoints to artifacts/vrsbench_detector_epoch{N}.pth
"""
import os, sys, io, json, time, zipfile
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms.functional as F

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
DEVICE = torch.device('cuda:0' if torch.cuda.device_count() == 1 else 'cuda:1')
torch.cuda.set_device(DEVICE)
print(f"Using device: {DEVICE} ({torch.cuda.get_device_name(DEVICE)})", flush=True)

ROOT = Path.home() / 'rsg-former'
SHARED = Path.home() / 'shared-assets/datasets/vrsbench'

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
        print(f"Dataset indexed {len(self.samples)} valid image-annotation pairs.", flush=True)

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

def main():
    img_zip = SHARED / 'Images_train.zip'
    ann_zip = SHARED / 'Annotations_train.zip'
    assert img_zip.exists(), f"Missing {img_zip}"
    assert ann_zip.exists(), f"Missing {ann_zip}"
    
    print(f"[{time.strftime('%X')}] Initializing VRSBench Full Object Detection Dataset...", flush=True)
    dataset = VRSBenchDetectionDataset(img_zip, ann_zip)
    
    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    print(f"[{time.strftime('%X')}] Building Faster R-CNN with ResNet-50-FPN (27 classes)...", flush=True)
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=len(CLASSES))
    model.to(DEVICE)
    model.train()
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4)
    EPOCHS = 5
    total_steps = EPOCHS * len(loader)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
    
    print(f"Starting training for {EPOCHS} epochs ({len(loader)} steps per epoch, batch_size=8)...", flush=True)
    
    epoch_metrics = []
    step_loss_log = []
    global_step = 0
    start_time = time.time()
    
    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        running_loss = 0.0
        running_cls = 0.0
        running_box = 0.0
        running_obj = 0.0
        running_rpn = 0.0
        trained_batches = 0
        
        for step, (images, targets) in enumerate(loader):
            valid_pairs = [(img, t) for img, t in zip(images, targets) if len(t['boxes']) > 0]
            if not valid_pairs:
                continue
                
            trained_batches += 1
            global_step += 1
            batch_imgs = [p[0].to(DEVICE) for p in valid_pairs]
            batch_tgts = [{k: v.to(DEVICE) for k, v in p[1].items()} for p in valid_pairs]
            
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                loss_dict = model(batch_imgs, batch_tgts)
                losses = sum(loss for loss in loss_dict.values())
                
            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
            optimizer.step()
            lr_scheduler.step()
            
            l_val = losses.item()
            running_loss += l_val
            running_cls += loss_dict.get('loss_classifier', torch.tensor(0.0)).item()
            running_box += loss_dict.get('loss_box_reg', torch.tensor(0.0)).item()
            running_obj += loss_dict.get('loss_objectness', torch.tensor(0.0)).item()
            running_rpn += loss_dict.get('loss_rpn_box_reg', torch.tensor(0.0)).item()
            
            if trained_batches % 100 == 0 or (step + 1) == len(loader):
                cur_lr = optimizer.param_groups[0]['lr']
                elapsed = (time.time() - start_time) / 60.0
                avg_l = running_loss / trained_batches
                pct = (step + 1) / len(loader) * 100.0
                eta_epoch = ((time.time() - epoch_start) / (step + 1)) * (len(loader) - step - 1) / 60.0
                print(f"  Epoch {epoch}/{EPOCHS} [{step+1}/{len(loader)} - {pct:.1f}%] | Loss: {avg_l:.4f} (cls: {running_cls/trained_batches:.4f}, box: {running_box/trained_batches:.4f}) | LR: {cur_lr:.2e} | Elapsed: {elapsed:.1f}m | ETA Epoch: {eta_epoch:.1f}m", flush=True)
                step_loss_log.append({
                    'epoch': epoch,
                    'step': step + 1,
                    'global_step': global_step,
                    'loss_total': l_val,
                    'loss_cls': loss_dict.get('loss_classifier', torch.tensor(0.0)).item(),
                    'loss_box': loss_dict.get('loss_box_reg', torch.tensor(0.0)).item(),
                    'lr': cur_lr,
                    'elapsed_min': elapsed
                })
                
        epoch_time = time.time() - epoch_start
        epoch_info = {
            'epoch': epoch,
            'loss_total_avg': running_loss / max(1, trained_batches),
            'loss_cls_avg': running_cls / max(1, trained_batches),
            'loss_box_avg': running_box / max(1, trained_batches),
            'loss_obj_avg': running_obj / max(1, trained_batches),
            'loss_rpn_avg': running_rpn / max(1, trained_batches),
            'epoch_seconds': epoch_time,
            'lr_end': optimizer.param_groups[0]['lr']
        }
        epoch_metrics.append(epoch_info)
        print(f"[Epoch {epoch} Complete] Avg Loss: {epoch_info['loss_total_avg']:.4f} | Time: {epoch_time/60.0:.1f}m", flush=True)
        
        ckpt_path = ROOT / f"artifacts/vrsbench_detector_epoch{epoch}.pth"
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch_metrics': epoch_info
        }, ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}", flush=True)
        
        out_metrics = ROOT / "reports/VRSBENCH_DETECTOR_TRAINING_METRICS.json"
        with open(out_metrics, 'w', encoding='utf-8') as f:
            json.dump({
                'task': 'VRSBench Object Detection Training (20,264 images)',
                'model': 'Faster R-CNN (ResNet-50-FPN)',
                'classes': CLASSES,
                'epochs_completed': epoch,
                'total_epochs': EPOCHS,
                'epoch_metrics': epoch_metrics
            }, f, indent=2)

        out_step = ROOT / "reports/VRSBENCH_DETECTOR_STEP_LOSS.json"
        with open(out_step, 'w', encoding='utf-8') as f:
            json.dump(step_loss_log, f, indent=2)
            
    print(f"[{time.strftime('%X')}] Training Finished! Total time: {(time.time()-start_time)/60.0:.1f}m", flush=True)

if __name__ == '__main__':
    main()
