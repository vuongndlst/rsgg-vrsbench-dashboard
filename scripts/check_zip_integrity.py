import subprocess
from pathlib import Path

target = Path.home() / 'shared-assets/datasets/coco_karpathy/train2014.zip'
print("File size:", target.stat().st_size if target.exists() else "Missing")

aria2_file = Path.home() / 'shared-assets/datasets/coco_karpathy/train2014.zip.aria2'
print("Aria2 control file exists:", aria2_file.exists())
if aria2_file.exists():
    print("Aria2 control file size:", aria2_file.stat().st_size)

# Check aria2c processes
print("\nAria2 processes:")
print(subprocess.getoutput("ps aux | grep aria2c | grep -v grep"))

# Check zip integrity with unzip -t (test only first few and last few entries or quick test)
print("\nTesting zip archive with zip -T or unzip -t...")
res = subprocess.run(f"unzip -t {target} | tail -n 10", shell=True, capture_output=True, text=True)
print("Returncode:", res.returncode)
print("Stdout:\n", res.stdout)
print("Stderr:\n", res.stderr)

