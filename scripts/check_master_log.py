from pathlib import Path
log_file = Path.home() / 'rsg-former' / 'artifacts' / 'multi_dataset_overnight.log'
if log_file.exists():
    print(log_file.read_text(encoding='utf-8'))
else:
    print("Log not found yet.")

