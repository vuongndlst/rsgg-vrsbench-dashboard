import os, subprocess
from pathlib import Path

target_dir = Path.home() / 'shared-assets/datasets/rsicd'
target_dir.mkdir(parents=True, exist_ok=True)

base_url = "https://huggingface.co/datasets/arampacha/rsicd/resolve/main/data"
files = [
    "test-00000-of-00001.parquet",
    "valid-00000-of-00001.parquet",
    "train-00000-of-00001.parquet"
]

for fname in files:
    out_file = target_dir / fname
    if not out_file.exists():
        url = f"{base_url}/{fname}"
        print(f"Downloading {url} to {out_file}...")
        cmd = f"aria2c -x 8 -s 8 -k 1M -c -d {target_dir} -o {fname} {url}"
        subprocess.run(cmd, shell=True, check=True)
        print(f"Downloaded {fname} ({out_file.stat().st_size / 1024**2:.2f} MB)")
    else:
        print(f"File {fname} already exists ({out_file.stat().st_size / 1024**2:.2f} MB)")

print("RSICD download completed successfully!")

