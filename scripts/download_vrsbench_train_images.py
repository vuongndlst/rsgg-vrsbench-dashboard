"""Download VRSBench Images_train.zip (7.97 GB, 20,264 images).
User requested: proceed without 34GB limit and keep COCO train intact.
"""
import os, sys, time, subprocess
from pathlib import Path

SHARED = Path.home() / 'shared-assets/datasets/vrsbench'
SHARED.mkdir(parents=True, exist_ok=True)
ROOT = Path.home() / 'rsg-former'
LOG_FILE = ROOT / 'artifacts/vrsbench_train_download.log'

def log(msg):
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(line + '\n')

train_zip = SHARED / 'Images_train.zip'
train_aria2 = SHARED / 'Images_train.zip.aria2'
train_url = "https://huggingface.co/datasets/xiang709/VRSBench/resolve/main/Images_train.zip"

log("=== Starting VRSBench Images_train.zip Download (User override: no 34GB limit) ===")
log(f"Current disk status:\n{subprocess.getoutput('df -h /home/stackops')}")

if train_zip.exists() and not train_aria2.exists() and train_zip.stat().st_size > 8_000_000_000:
    log(f"Images_train.zip already downloaded ({train_zip.stat().st_size / (1024*1024):.2f} MB)!")
else:
    log("Downloading Images_train.zip (7.97 GB, 20,264 training images) via aria2c...")
    cmd = (
        f"aria2c -x 16 -s 16 -j 16 -k 1M -c --summary-interval=15 "
        f"-d {SHARED} -o Images_train.zip {train_url}"
    )
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in iter(p.stdout.readline, ''):
        line = line.strip()
        if 'DL:' in line or 'complete' in line or 'Results:' in line or 'OK' in line or 'ETA' in line:
            log(f"  [train] {line}")
    p.wait()
    if p.returncode != 0:
        log(f"ERROR: aria2c failed with returncode {p.returncode}")
        sys.exit(p.returncode)
    log(f"Images_train.zip download complete! Size: {train_zip.stat().st_size / (1024*1024):.2f} MB")

log(f"Final disk status:\n{subprocess.getoutput('df -h /home/stackops')}")
log("=== VRSBench Images_train.zip Download Finished ===")

