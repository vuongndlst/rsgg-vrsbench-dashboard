import subprocess
from pathlib import Path

hub_dir = Path.home() / ".cache/huggingface/hub/.locks/models--Qwen--Qwen2.5-14B-Instruct"
for p in hub_dir.glob("*.lock"):
    out = subprocess.getoutput(f"fuser {p}")
    print(p.name, out)

