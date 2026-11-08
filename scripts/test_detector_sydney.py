import os, sys, torch
from pathlib import Path
from PIL import Image
from torchvision.transforms import functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

ROOT = Path.home() / 'rsg-former'
ckpt_path = ROOT / 'artifacts' / 'vrsbench_detector_v2_best.pth'
ckpt = torch.load(ckpt_path, map_location='cpu')

CLASSES = ckpt['classes']
model = fasterrcnn_resnet50_fpn_v2(weights=None)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(CLASSES))

model.load_state_dict(ckpt['model_state_dict'])
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()
print(f"Detector v2 loaded successfully on {device} (Epoch {ckpt.get('epoch')})")

img_dir = ROOT / 'data' / 'sydney_captions' / 'images'
sydney_imgs = sorted(list(img_dir.glob("*.tif")))[:6]

with torch.no_grad():
    for p in sydney_imgs:
        img = Image.open(p).convert('RGB')
        t = F.to_tensor(img).unsqueeze(0).to(device)
        preds = model(t)[0]
        boxes = preds['boxes'].cpu().numpy()
        scores = preds['scores'].cpu().numpy()
        labels = preds['labels'].cpu().numpy()
        
        keep = scores >= 0.20
        dets = []
        for b, s, l in zip(boxes[keep], scores[keep], labels[keep]):
            dets.append(f"{CLASSES[l]} ({s:.2f})")
        print(f"Image {p.name}: {len(dets)} detections -> {dets[:6]}")
