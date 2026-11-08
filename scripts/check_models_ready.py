import subprocess
from pathlib import Path

hub_dir = Path.home() / ".cache/huggingface/hub"
for m in ["models--Qwen--Qwen2.5-7B-Instruct", "models--Qwen--Qwen2.5-14B-Instruct"]:
    p = hub_dir / m
    if p.exists():
        size_mb = sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / (1024*1024)
        print(f"{m}: {size_mb:.1f} MB")
        incompletes = list(p.rglob("*.incomplete"))
        print(f"  incomplete files: {len(incompletes)}")
        snapshots = list((p / "snapshots").glob("*"))
        print(f"  snapshots: {[s.name for s in snapshots]}")

