import os
import glob
import subprocess
import json

print("=== GPU STATUS ===")
try:
    smi = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
    print(smi.strip())
except Exception as e:
    print("nvidia-smi error:", e)

print("\n=== RUNNING PYTHON PROCESSES ===")
try:
    ps = subprocess.check_output(["ps", "aux"], text=True)
    for line in ps.splitlines():
        if "python" in line and not "check_live_status.py" in line:
            print(line[:120])
except Exception as e:
    print("ps error:", e)

print("\n=== REPORTS DIRECTORY ===")
for p in sorted(glob.glob("reports/*")):
    size_mb = os.path.getsize(p) / (1024 * 1024)
    print(f"{p}: {size_mb:.2f} MB")

