import torch
from pathlib import Path

ckpt_path = Path.home() / 'rsg-former' / 'artifacts' / 'vrsbench_detector_v2_best.pth'
d = torch.load(ckpt_path, map_location='cpu')
print("Keys:", d.keys())
print("Epoch:", d.get('epoch'))
print("Classes:", d.get('classes'))
print("Metrics:", d.get('metrics'))

