"""Download VRSBench image archives (Images_val.zip and Images_train.zip) safely with aria2c.
Strict safety rule: Maintain >= 34 GB free disk space on /home/stackops.
"""
import os, sys, time, json, subprocess
from pathlib import Path

SHARED = Path.home() / 'shared-assets/datasets/vrsbench'
SHARED.mkdir(parents=True, exist_ok=True)
ROOT = Path.home() / 'rsg-former'
LOG_FILE = ROOT / 'artifacts/vrsbench_image_download.log'

def get_free_mb():
    stat = os.statvfs(str(Path.home()))
    return (stat.f_bavail * stat.f_frsize) // (1024 * 1024)

def log(msg):
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(line + '\n')

log("=== Starting VRSBench Image Download Pipeline ===")
free_mb = get_free_mb()
log(f"Current free disk space: {free_mb / 1024:.2f} GB ({free_mb} MB)")

if free_mb < 34000:
    log(f"ERROR: Free disk space {free_mb} MB is below safety threshold 34000 MB! Aborting.")
    sys.exit(1)

# Step 1: Download Images_val.zip (3.79 GB - 9,350 evaluation images)
val_zip = SHARED / 'Images_val.zip'
val_url = "https://huggingface.co/datasets/xiang709/VRSBench/resolve/main/Images_val.zip"
val_aria2 = SHARED / 'Images_val.zip.aria2'

if val_zip.exists() and not val_aria2.exists() and val_zip.stat().st_size > 3_900_000_000:
    log(f"Images_val.zip already downloaded ({val_zip.stat().st_size / (1024*1024):.2f} MB)!")
else:
    log("Downloading Images_val.zip (3.79 GB, 9,350 evaluation images)...")
    cmd = (
        f"aria2c -x 16 -s 16 -j 16 -k 1M -c --summary-interval=10 "
        f"-d {SHARED} -o Images_val.zip {val_url}"
    )
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in iter(p.stdout.readline, ''):
        line = line.strip()
        if 'DL:' in line or 'complete' in line or 'Results:' in line or 'OK' in line or 'ETA' in line:
            log(f"  [val] {line}")
    p.wait()
    if p.returncode != 0:
        log(f"ERROR: aria2c failed with returncode {p.returncode}")
        sys.exit(p.returncode)
    log(f"Images_val.zip download complete! Size: {val_zip.stat().st_size / (1024*1024):.2f} MB")

# Check free space after val download
free_mb = get_free_mb()
log(f"Free disk space after Images_val.zip: {free_mb / 1024:.2f} GB ({free_mb} MB)")

# Step 2: Download Images_train.zip (7.97 GB) only if safe (>= 34 GB threshold)
train_zip = SHARED / 'Images_train.zip'
train_url = "https://huggingface.co/datasets/xiang709/VRSBench/resolve/main/Images_train.zip"
train_aria2 = SHARED / 'Images_train.zip.aria2'

if train_zip.exists() and not train_aria2.exists() and train_zip.stat().st_size > 8_000_000_000:
    log(f"Images_train.zip already downloaded ({train_zip.stat().st_size / (1024*1024):.2f} MB)!")
else:
    # Need at least 8 GB + 34 GB = 42 GB free to download without dropping below 34 GB
    if free_mb < (34000 + 8500):
        log(f"NOTICE: Free space is {free_mb / 1024:.2f} GB. Downloading Images_train.zip (8.0 GB) now would temporarily leave { (free_mb - 8500) / 1024:.2f} GB, which is below the 34 GB safety threshold while COCO download is also present.")
        log("Waiting for COCO train2014 cleanup or user decision before starting Images_train.zip.")
    else:
        log("Downloading Images_train.zip (7.97 GB, 20,264 training images)...")
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

log("=== VRSBench Image Download Step Finished ===")
log(f"Final free disk space: {get_free_mb() / 1024:.2f} GB")

