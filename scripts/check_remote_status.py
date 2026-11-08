import os, subprocess, json
from pathlib import Path

def run_cmd(cmd):
    return subprocess.getoutput(cmd)

print("=== DISK SPACE ===")
print(run_cmd("df -h /"))

print("\n=== GPU STATUS ===")
print(run_cmd("nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader"))

print("\n=== PYTHON / ARIA2 PROCESSES ===")
print(run_cmd("ps aux | grep -E 'python|aria2' | grep -v grep"))

print("\n=== CHECK REPORTS ===")
rep = Path.home() / "rsg-former/reports/VRSBENCH_QWEN_G2T_FULL_TRAIN_REPORT.json"
if rep.exists():
    print(f"Report exists! Size: {rep.stat().st_size} bytes")
    try:
        with open(rep) as f:
            d = json.load(f)
        print("Final Report content:")
        print(json.dumps(d, indent=2))
    except Exception as e:
        print("Error reading report:", e)
else:
    print(f"Report file {rep} not found yet.")

print("\n=== CHECK ARTIFACTS DIR ===")
art_dir = Path.home() / "rsg-former/artifacts"
if art_dir.exists():
    for p in art_dir.glob("qwen_vrsbench*"):
        print(f"Artifact: {p.name}")

print("\n=== CHECK IMAGES_TRAIN.ZIP ===")
train_zip = Path.home() / "shared-assets/datasets/vrsbench/Images_train.zip"
if train_zip.exists():
    print(f"Images_train.zip exists! Size: {train_zip.stat().st_size / (1024**3):.2f} GB")
aria_file = Path.home() / "shared-assets/datasets/vrsbench/Images_train.zip.aria2"
if aria_file.exists():
    print(f"Images_train.zip.aria2 exists (download in progress / paused).")
