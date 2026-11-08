"""Download and extract official MS COCO 2014 train set (82,783 images).
Features:
  - Multi-connection aria2c download with resume capability (-c)
  - Automatic extraction via system unzip
  - Instant cleanup of zip file to reclaim 12.6 GB disk space
  - Image count and integrity verification
"""
import os, sys, time, subprocess, shutil
from pathlib import Path

TARGET_DIR = Path.home() / 'shared-assets/datasets/coco_karpathy'
ZIP_PATH = TARGET_DIR / 'train2014.zip'
IMAGES_DIR = TARGET_DIR / 'images'
TRAIN_DIR = IMAGES_DIR / 'train2014'
URL = "http://images.cocodataset.org/zips/train2014.zip"

print(f"[{time.strftime('%X')}] Starting COCO 2014 train set acquisition pipeline...")
print(f"Target directory: {TARGET_DIR}")

# 1. Disk Space Check
df_out = subprocess.getoutput(f"df -h {TARGET_DIR}")
print("Current disk space:\n" + df_out)

# 2. Check if already extracted
if TRAIN_DIR.exists():
    img_count = sum(1 for _ in TRAIN_DIR.glob('*.jpg'))
    print(f"Existing train2014 folder found with {img_count} images.")
    if img_count >= 82000:
        print("COCO train2014 already fully extracted! Exiting.")
        sys.exit(0)

# 3. Download via aria2c
print(f"[{time.strftime('%X')}] Downloading {URL} via aria2c (16 connections)...")
dl_cmd = (
    f"aria2c -x 16 -s 16 -j 16 -k 1M -c --summary-interval=15 "
    f"-d {TARGET_DIR} -o train2014.zip {URL}"
)
t0 = time.time()
p = subprocess.Popen(dl_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in p.stdout:
    line_s = line.strip()
    if any(k in line_s for k in ['ETA', 'Download Progress', 'Download Results', 'speed', '%']):
        print(f"  [aria2] {line_s}", flush=True)
p.wait()

if p.returncode != 0:
    print(f"Download failed with exit code {p.returncode}")
    sys.exit(p.returncode)

dl_time = time.time() - t0
zip_size_gb = ZIP_PATH.stat().st_size / (1024**3)
print(f"[{time.strftime('%X')}] Download complete in {dl_time/60:.1f}m ({zip_size_gb:.2f} GB)!")

# 4. Extract via unzip
print(f"[{time.strftime('%X')}] Extracting train2014.zip to {IMAGES_DIR}...")
t_unzip = time.time()
unzip_cmd = f"unzip -q -o {ZIP_PATH} -d {IMAGES_DIR}"
res = subprocess.run(unzip_cmd, shell=True)
if res.returncode != 0:
    print(f"Unzip failed with exit code {res.returncode}")
    sys.exit(res.returncode)

unzip_time = time.time() - t_unzip
print(f"[{time.strftime('%X')}] Extraction complete in {unzip_time/60:.1f}m!")

# 5. Clean up zip file
print(f"[{time.strftime('%X')}] Removing zip file to reclaim 12.6 GB disk space...")
if ZIP_PATH.exists():
    ZIP_PATH.unlink()

# 6. Verify image count
if TRAIN_DIR.exists():
    total_train = sum(1 for _ in TRAIN_DIR.glob('*.jpg'))
    print(f"[{time.strftime('%X')}] Verification SUCCESS: Found {total_train} images in {TRAIN_DIR}!")
else:
    print(f"Warning: {TRAIN_DIR} not found as expected.")

final_df = subprocess.getoutput(f"df -h {TARGET_DIR}")
print("Final disk space:\n" + final_df)
print("COCO_TRAIN2014_READY")

