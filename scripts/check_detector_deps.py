import torch
import torchvision
print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
try:
    import timm
    print("timm version:", timm.__version__)
except ImportError:
    print("timm not installed")

try:
    import pycocotools
    print("pycocotools is installed")
except ImportError:
    print("pycocotools not installed")

