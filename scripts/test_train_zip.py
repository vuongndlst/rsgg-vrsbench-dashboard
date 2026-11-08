import subprocess
from pathlib import Path

train_zip = Path.home() / "shared-assets/datasets/vrsbench/Images_train.zip"
print(f"Testing {train_zip} ({train_zip.stat().st_size / (1024**3):.2f} GB)...")
res = subprocess.run(["unzip", "-t", str(train_zip)], capture_output=True, text=True)
print("Return code:", res.returncode)
lines = res.stdout.strip().split("\n")
print("First 3 lines:\n", "\n".join(lines[:3]))
print("Last 3 lines:\n", "\n".join(lines[-3:]))
if res.returncode == 0:
    print("SUCCESS: Images_train.zip is 100% COMPLETE AND VALID!")
else:
    print("ERROR testing zip:\n", res.stderr[-500:])

