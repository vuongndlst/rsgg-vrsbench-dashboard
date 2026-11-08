from pathlib import Path
log_p = Path.home() / 'rsg-former' / 'artifacts' / 'vrsbench_detector_v2_train.log'
if log_p.exists():
    text = log_p.read_text(encoding='utf-8', errors='replace')
    print("Log size:", len(text))
    print("Tail:\n" + text[-500:])

