import sys, time
from pathlib import Path
import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())

try:
    import bitsandbytes as bnb
    print("bitsandbytes version:", bnb.__version__)
except Exception as e:
    print("bitsandbytes import error:", e)

from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "Qwen/Qwen2.5-7B-Instruct"
print(f"Testing tokenizer load for {model_name}...")
t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(model_name)
print(f"Tokenizer loaded in {time.time()-t0:.2f}s")

