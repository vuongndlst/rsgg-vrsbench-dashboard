import os
from pathlib import Path

log_file = Path.home() / "rsg-former/artifacts/multi_dataset_overnight.log"
if log_file.exists():
    print(f"Log size: {log_file.stat().st_size} bytes")
    print(log_file.read_text(encoding="utf-8", errors="replace"))
else:
    print("Log file does not exist.")

