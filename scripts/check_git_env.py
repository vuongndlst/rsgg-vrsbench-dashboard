import sys, os
from pathlib import Path

ROOT = Path.home() / 'rsg-former'
sys.path.insert(0, str(ROOT / '.tools/deps'))
sys.path.insert(0, str(ROOT / 'src'))

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.cider.cider import Cider

print("All imports in project environment OK!")
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device name:", torch.cuda.get_device_name(0))

