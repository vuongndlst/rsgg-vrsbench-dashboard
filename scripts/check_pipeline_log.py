import time
from pathlib import Path

ROOT = Path.home() / 'rsg-former'
log_file = ROOT / 'artifacts/overnight_pipeline.log'
time.sleep(2)
if log_file.exists():
    lines = log_file.read_text(encoding='utf-8', errors='replace').splitlines()
    print("OVERNIGHT PIPELINE LOG (last 20 lines):")
    for l in lines[-20:]:
        print(l)
else:
    print("Log file not found!")

