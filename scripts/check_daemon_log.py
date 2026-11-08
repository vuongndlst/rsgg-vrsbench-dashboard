from pathlib import Path
log_file = Path.home() / 'rsg-former/artifacts/coco_daemon.log'
if log_file.exists():
    print(log_file.read_text())
else:
    print("Log file not found yet.")

