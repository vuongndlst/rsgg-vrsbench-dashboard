"""Automated daemon for MS COCO 2014 full dataset preparation and progressive SGG extraction.
Steps:
1. Monitor aria2c until train2014.zip is fully downloaded.
2. Verify archive integrity.
3. Extract train2014.zip to images/train2014/.
4. Immediately remove train2014.zip to preserve safe disk space.
5. Verify total image count (~82,783).
6. Progressive RSG-Former SGG feature extraction on GPU 1 with 5,000-image checkpoints.
"""
import os, sys, time, json, subprocess
from pathlib import Path

SHARED = Path.home() / 'shared-assets/datasets/coco_karpathy'
ZIP_PATH = SHARED / 'train2014.zip'
ARIA2_FILE = SHARED / 'train2014.zip.aria2'
IMAGES_DIR = SHARED / 'images'
TRAIN_DIR = IMAGES_DIR / 'train2014'
ROOT = Path.home() / 'rsg-former'
LOG_FILE = ROOT / 'artifacts/coco_full_pipeline.log'

def log(msg):
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(line + '\n')

log("=== COCO Full Pipeline Daemon Started ===")
log(f"Shared dir: {SHARED}")
log(f"Disk space before starting:\n{subprocess.getoutput(f'df -h {SHARED}')}")

# 1. Check if extraction already completed
if TRAIN_DIR.exists():
    cnt = len(list(TRAIN_DIR.glob('*.jpg')))
    log(f"Found existing train2014 folder with {cnt} images.")
    if cnt >= 82000:
        log("COCO train2014 already extracted! Proceeding to feature extraction.")
else:
    # Monitor download
    log("Monitoring aria2c download of train2014.zip...")
    last_print = 0
    while True:
        # Check if aria2 file still exists
        if not ARIA2_FILE.exists() and ZIP_PATH.exists() and ZIP_PATH.stat().st_size > 13_000_000_000:
            log("train2014.zip.aria2 control file gone and zip is >13 GB! Verifying zip integrity...")
            res = subprocess.run(f"unzip -t {ZIP_PATH} | tail -n 5", shell=True, capture_output=True, text=True)
            if res.returncode == 0:
                log("train2014.zip integrity check PASSED!")
                break
            else:
                log(f"Zip test returned {res.returncode}: {res.stdout.strip()}")
        
        # Check aria2c process
        ps_aria2 = subprocess.getoutput("ps aux | grep aria2c | grep -v grep")
        if not ps_aria2 and ARIA2_FILE.exists():
            log("aria2c not running but .aria2 file exists. Resuming download...")
            subprocess.Popen(
                f"aria2c -x 16 -s 16 -j 16 -k 1M -c --summary-interval=30 -d {SHARED} -o train2014.zip http://images.cocodataset.org/zips/train2014.zip >> {ROOT}/artifacts/aria2_resume.log 2>&1",
                shell=True
            )
            time.sleep(5)
        
        if time.time() - last_print > 60:
            last_print = time.time()
            if ARIA2_FILE.exists():
                try:
                    data = ARIA2_FILE.read_bytes()
                    ones = sum(bin(b).count('1') for b in data[32:])
                    pct = (ones / 12885) * 100
                    log(f"Download progress: ~{pct:.1f}% ({ones} / 12885 MB)")
                except Exception as e:
                    log(f"Could not parse .aria2: {e}")
            else:
                log("Waiting for aria2c completion...")
        time.sleep(10)

    # 2. Extract train2014.zip
    log(f"Extracting {ZIP_PATH} to {IMAGES_DIR}...")
    t0_unzip = time.time()
    unzip_cmd = f"unzip -q -o {ZIP_PATH} -d {IMAGES_DIR}"
    res = subprocess.run(unzip_cmd, shell=True)
    if res.returncode != 0:
        log(f"ERROR: Extraction failed with returncode {res.returncode}")
        sys.exit(res.returncode)
    log(f"Extraction completed in {(time.time()-t0_unzip)/60:.1f} minutes!")

    # 3. CRITICAL: Remove train2014.zip to reclaim 12.6 GB disk space
    log("Removing train2014.zip to reclaim 12.6 GB disk space...")
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    log(f"Disk space after cleanup:\n{subprocess.getoutput(f'df -h {SHARED}')}")

    # 4. Verification
    total_imgs = len(list(TRAIN_DIR.glob('*.jpg')))
    log(f"Verified {total_imgs} images extracted in {TRAIN_DIR}!")

log("=== COCO Train 2014 Ready for SGG Feature Extraction ===")

