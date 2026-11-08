import time
from pathlib import Path

ROOT = Path.home() / 'rsg-former'
log_file = ROOT / 'artifacts/vrsbench_detector_v2_train.log'
time.sleep(2)
if log_file.exists():
    lines = log_file.read_text(encoding='utf-8', errors='replace').splitlines()
    print("LOG TAIL (last 25 lines):")
    for l in lines[-25:]:
        print(l)
else:
    print("Log file not found!")

