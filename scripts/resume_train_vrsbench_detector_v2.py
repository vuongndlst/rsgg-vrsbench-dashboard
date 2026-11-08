"""Resume Stage 2 SOTA Object Detector Training on VRSBench for Epoch 4 & 5:
Resumes from artifacts/vrsbench_detector_v2_epoch3.pth
Runs on GPU 1 (which has 24GB free VRAM).
Completely immune to SSH disconnect:
- Disables SIGPIPE
- Flushes directly to artifacts/vrsbench_detector_v2_train.log
- Saves checkpoints at mid-epoch and end of each epoch
"""
import os, sys, io, json, time, zipfile, random, signal
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms.functional as F

# Ignore SIGPIPE completely
try:
    signal.signal(signal.SIGPIPE, signal.SIG_IGN)
except Exception:
    pass

DEVICE = torch.device('cuda:0')
torch.cuda.set_device(DEVICE)

ROOT = Path.home() / 'rsg-former'
SHARED = Path.home() / 'shared-assets/datasets/vrsbench'
LOG_FILE = ROOT / 'artifacts' / 'vrsbench_detector_v2_train.log'

class DualLogger:
    def __init__(self, filename):
        self.file = open(filename, 'a', buffering=1, encoding='utf-8')
        self.stdout = sys.__stdout__

    def write(self, msg):
        try:
            self.file.write(msg)
            self.file.flush()
        except Exception:
            pass
        try:
            self.stdout.write(msg)
            self.stdout.flush()
        except Exception:
            pass

    def flush(self):
        try:
            self.file.flush()
        except Exception:
            pass
        try:
            self.stdout.flush()
        except Exception:
            pass

sys.stdout = DualLogger(LOG_FILE)
sys.stderr = sys.stdout

CLASSES = [
    "background",
    "vehicle", "airplane", "ship", "harbor", "bridge", "tennis-court",
    "storage-tank", "baseball-diamond", "swimming-pool", "ground-track-field",
    "basketball-court", "roundabout", "windmill", "overpass", "soccer-ball-field",
    "expressway-service-area", "airport", "trainstation", "expressway-toll-station",
    "chimney", "stadium", "dam", "golffield", "helicopter", "container-crane", "helipad"
]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}

class VRSBenchDetectionDatasetV2(Dataset):
    def __init__(self, img_zip_path, ann_zip_path, augment=True):
        self.img_zip_path = str(img_zip_path)
        self.ann_zip_path = str(ann_zip_path)
        self.augment = augment
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
        print(f"Dataset indexed {len(self.samples)} valid image-annotation pairs (augment={augment}).", flush=True)

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
                    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                else:
                    continue

            x1, y1, x2, y2 = max(0.0, x1), max(0.0, y1), min(float(w), x2), min(float(h), y2)
            if x2 - x1 >= 2.0 and y2 - y1 >= 2.0:
                boxes.append([x1, y1, x2, y2])
                labels.append(CLASS_TO_IDX[cls_name])

        if self.augment and len(boxes) > 0:
            if random.random() < 0.5:
                img_tensor = torch.flip(img_tensor, [2])
                new_boxes = []
                for b in boxes:
                    new_boxes.append([w - b[2], b[1], w - b[0], b[3]])
                boxes = new_boxes
            if random.random() < 0.5:
                img_tensor = torch.flip(img_tensor, [1])
                new_boxes = []
                for b in boxes:
                    new_boxes.append([b[0], h - b[3], b[2], h - b[1]])
                boxes = new_boxes

        if len(boxes) == 0:
            target = {
                'boxes': torch.zeros((0, 4), dtype=torch.float32),
                'labels': torch.zeros((0,), dtype=torch.int64),
                'image_id': torch.tensor([idx])
            }
        else:
            target = {
                'boxes': torch.as_tensor(boxes, dtype=torch.float32),
                'labels': torch.as_tensor(labels, dtype=torch.int64),
                'image_id': torch.tensor([idx])
            }
        return img_tensor, target

def collate_fn(batch):
    return tuple(zip(*batch))

def compute_iou(b1, b2):
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    a2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    u = a1 + a2 - inter
    return inter / u if u > 0 else 0.0

def evaluate_val_subset(model, val_samples, val_img_zip):
    model.eval()
    ious_ag = []
    ious_cl = []

    with zipfile.ZipFile(val_img_zip) as iz:
        img_names = set(iz.namelist())
        with torch.inference_mode(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            for s in val_samples:
                img_id = s['image_id']
                img_cand = f"Images_val/{img_id}" if f"Images_val/{img_id}" in img_names else img_id
                raw_bytes = iz.read(img_cand)
                img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
                w, h = img.size
                t = F.to_tensor(img).to(DEVICE)
                preds = model([t])[0]

                p_boxes = preds['boxes'].cpu().tolist()
                p_labels = preds['labels'].cpu().tolist()
                p_scores = preds['scores'].cpu().tolist()

                det_objs = []
                for b, l, sc in zip(p_boxes, p_labels, p_scores):
                    if sc < 0.10:
                        continue
                    cname = IDX_TO_CLASS.get(int(l), 'background')
                    if cname == 'background':
                        continue
                    det_objs.append({
                        'obj_cls': cname,
                        'coord': [b[0]/w, b[1]/h, b[2]/w, b[3]/h],
                        'score': sc
                    })

                for gt in s['gt_objects']:
                    gcls = gt['obj_cls']
                    gbox = gt['obj_coord']
                    best_ag = max([compute_iou(gbox, d['coord']) for d in det_objs], default=0.0)
                    best_cl = max([compute_iou(gbox, d['coord']) for d in det_objs if d['obj_cls'] == gcls], default=0.0)
                    ious_ag.append(best_ag)
                    ious_cl.append(best_cl)

    model.train()
    a_ag = np.array(ious_ag)
    a_cl = np.array(ious_cl)
    return {
        'val_ag_mIoU': round(float(np.mean(a_ag)*100), 2),
        'val_ag_Acc50': round(float(np.mean(a_ag>=0.5)*100), 2),
        'val_ag_Acc70': round(float(np.mean(a_ag>=0.7)*100), 2),
        'val_cl_mIoU': round(float(np.mean(a_cl)*100), 2),
        'val_cl_Acc50': round(float(np.mean(a_cl>=0.5)*100), 2),
        'val_cl_Acc70': round(float(np.mean(a_cl>=0.7)*100), 2),
    }

def main():
    img_zip = SHARED / 'Images_train.zip'
    ann_zip = SHARED / 'Annotations_train.zip'
    val_cap_file = SHARED / 'VRSBench_EVAL_Cap.json'
    val_ann_zip = SHARED / 'Annotations_val.zip'
    val_img_zip = SHARED / 'Images_val.zip'

    cap_data = json.loads(val_cap_file.read_text(encoding='utf-8'))[:200]
    val_samples = []
    with zipfile.ZipFile(val_ann_zip) as az:
        ann_names = set(az.namelist())
        for item in cap_data:
            img_id = item['image_id']
            stem = Path(img_id).stem
            ann_cand = f"Annotations_val/{stem}.json" if f"Annotations_val/{stem}.json" in ann_names else f"{stem}.json"
            gt_objs = []
            if ann_cand in ann_names:
                adata = json.loads(az.read(ann_cand).decode('utf-8'))
                for o in adata.get('objects', []):
                    if len(o.get('obj_coord', [])) == 4:
                        gt_objs.append({'obj_cls': o.get('obj_cls', ''), 'obj_coord': o['obj_coord']})
            val_samples.append({'image_id': img_id, 'gt_objects': gt_objs})

    dataset = VRSBenchDetectionDatasetV2(img_zip, ann_zip, augment=True)
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=False
    )

    print(f"[{time.strftime('%X')}] Loading Faster R-CNN ResNet-50-FPN-v2 from epoch 3 checkpoint...", flush=True)
    model = fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=len(CLASSES))
    model.transform.min_size = (800, 864, 928)
    model.transform.max_size = 1333

    ckpt_path = ROOT / 'artifacts/vrsbench_detector_v2_epoch3.pth'
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"Loaded epoch 3 state dict successfully! (Previous metrics: {ckpt.get('metrics')})", flush=True)

    model.to(DEVICE)
    model.train()

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=6.0e-5, weight_decay=1e-4) # Annealed LR for epoch 4-5
    EPOCHS = 5
    START_EPOCH = 4
    GRAD_ACCUM = 4
    total_optim_steps = (2 * len(loader)) // GRAD_ACCUM
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_optim_steps, eta_min=1e-6)

    out_dir = ROOT / 'artifacts'
    rep_dir = ROOT / 'reports'
    out_dir.mkdir(parents=True, exist_ok=True)
    rep_dir.mkdir(parents=True, exist_ok=True)

    # Load existing history
    metrics_file = rep_dir / 'VRSBENCH_DETECTOR_V2_TRAINING_METRICS.json'
    best_miou = 70.28
    epoch_metrics = []
    if metrics_file.exists():
        try:
            prev = json.loads(metrics_file.read_text(encoding='utf-8'))
            epoch_metrics = prev.get('epoch_history', [])
            best_miou = prev.get('best_val_mIoU', best_miou)
        except Exception:
            pass

    start_time = time.time()
    for epoch in range(START_EPOCH, EPOCHS + 1):
        epoch_start = time.time()
        running_loss = 0.0
        trained_batches = 0
        optimizer.zero_grad()

        for step, (images, targets) in enumerate(loader):
            valid_pairs = [(img, t) for img, t in zip(images, targets) if len(t['boxes']) > 0]
            if not valid_pairs:
                continue

            trained_batches += 1
            batch_imgs = [p[0].to(DEVICE) for p in valid_pairs]
            batch_tgts = [{k: v.to(DEVICE) for k, v in p[1].items()} for p in valid_pairs]

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                loss_dict = model(batch_imgs, batch_tgts)
                losses = sum(loss for loss in loss_dict.values()) / GRAD_ACCUM

            losses.backward()

            if trained_batches % GRAD_ACCUM == 0 or (step + 1) == len(loader):
                torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

            loss_val = float(losses.item() * GRAD_ACCUM)
            running_loss += loss_val

            if trained_batches % 400 == 0 or (step + 1) == len(loader):
                avg_l = running_loss / max(1, trained_batches)
                elapsed_m = (time.time() - start_time) / 60.0
                print(
                    f"Epoch [{epoch}/{EPOCHS}] Step [{step+1}/{len(loader)}] (Valid: {trained_batches}) | "
                    f"Loss: {loss_val:.4f} (Avg: {avg_l:.4f}) | "
                    f"LR: {optimizer.param_groups[0]['lr']:.2e} | Elapsed: {elapsed_m:.1f}m",
                    flush=True
                )

            if trained_batches % 3000 == 0:
                val_mid = evaluate_val_subset(model, val_samples, val_img_zip)
                print(f"  [Mid-Epoch {epoch} @ {trained_batches} batches] Val: {json.dumps(val_mid)}", flush=True)
                if val_mid['val_ag_mIoU'] >= best_miou:
                    best_miou = val_mid['val_ag_mIoU']
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'classes': CLASSES,
                        'metrics': val_mid
                    }, out_dir / "vrsbench_detector_v2_best.pth")

        val_res = evaluate_val_subset(model, val_samples, val_img_zip)
        ep_sec = time.time() - epoch_start
        ep_summary = {
            'epoch': epoch,
            'loss_total_avg': round(running_loss / max(1, trained_batches), 4),
            'epoch_seconds': round(ep_sec, 1),
            **val_res
        }
        epoch_metrics.append(ep_summary)
        print(f"=== Epoch {epoch} Complete ({ep_sec/60:.1f}m) | Val: {json.dumps(val_res)} ===", flush=True)

        ckpt_file = out_dir / f"vrsbench_detector_v2_epoch{epoch}.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'classes': CLASSES,
            'metrics': ep_summary
        }, ckpt_file)

        if val_res['val_ag_mIoU'] >= best_miou:
            best_miou = val_res['val_ag_mIoU']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'classes': CLASSES,
                'metrics': ep_summary
            }, out_dir / "vrsbench_detector_v2_best.pth")

        metrics_file.write_text(
            json.dumps({
                'model': 'Faster R-CNN ResNet-50-FPN-v2 (Pretrained COCO_V1 + High-Res Aerial Augmentation)',
                'epochs_completed': epoch,
                'total_epochs': EPOCHS,
                'best_val_mIoU': best_miou,
                'epoch_history': epoch_metrics
            }, indent=2),
            encoding='utf-8'
        )

if __name__ == '__main__':
    main()

