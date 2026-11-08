import subprocess
from pathlib import Path

ROOT = Path.home() / 'rsg-former'
log_file = ROOT / 'artifacts/coco_daemon.log'
cmd = f"nohup /usr/bin/python3 -u {ROOT}/scripts/coco_full_pipeline_daemon.py > {log_file} 2>&1 &"

print(f"Launching daemon: {cmd}")
subprocess.Popen(cmd, shell=True)
print("Daemon launched successfully!")

