import os, json, time, subprocess
from pathlib import Path

ROOT = Path.home() / 'rsg-former'
train_log = ROOT / 'artifacts' / 'vrsbench_detector_v2_train.log'
if train_log.exists():
    lines = train_log.read_text(encoding='utf-8', errors='replace').splitlines()
    non_empty = [l.strip() for l in lines if l.strip()]
    print("Latest training lines:")
    for l in non_empty[-15:]:
        print(" ", l[:120])
else:
    print("Log not found.")
