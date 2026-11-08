import sys, time
from pathlib import Path
sys.path.insert(0, str(Path.home() / 'rsg-former/scripts'))
import torch

from train_vrsbench_detector import VRSBenchDetectionDataset, collate_fn, CLASSES, CLASS_TO_IDX
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

SHARED = Path.home() / 'shared-assets/datasets/vrsbench'
img_zip = SHARED / 'Images_train.zip'
ann_zip = SHARED / 'Annotations_train.zip'

print("Initializing test dataset...")
ds = VRSBenchDetectionDataset(img_zip, ann_zip)
print(f"Total samples: {len(ds)}")

# Find a sample with boxes
sample_idx = 0
for i in range(10):
    img, target = ds[i]
    if len(target['boxes']) > 0:
        sample_idx = i
        break

img, target = ds[sample_idx]
print(f"Sample {sample_idx}: img shape={img.shape}, num boxes={len(target['boxes'])}, labels={target['labels']}")

DEVICE = torch.device('cuda:0')
images = [img.to(DEVICE)]
targets = [{k: v.to(DEVICE) for k, v in target.items()}]

print("Initializing model...")
model = fasterrcnn_resnet50_fpn(weights=None)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=len(CLASSES))
model.to(DEVICE)
model.train()

print("Running test forward-backward step...")
with torch.cuda.amp.autocast(dtype=torch.bfloat16):
    loss_dict = model(images, targets)
    loss = sum(l for l in loss_dict.values())

loss.backward()
print(f"SUCCESS! Test forward & backward passed. Loss = {loss.item():.4f}")
for k, v in loss_dict.items():
    print(f"  {k}: {v.item():.4f}")
