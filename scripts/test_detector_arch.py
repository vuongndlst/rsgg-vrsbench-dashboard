import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

print("Creating Faster R-CNN with ResNet50-FPN...")
model = fasterrcnn_resnet50_fpn(weights=None)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=27)
print("Model created successfully with 27 classes (26 RS classes + background)!")
params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {params/1e6:.2f} M")

