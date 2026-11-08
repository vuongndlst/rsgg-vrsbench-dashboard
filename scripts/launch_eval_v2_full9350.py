import subprocess, os, sys
from pathlib import Path

ROOT = Path.home() / 'rsg-former'
log_file = ROOT / 'reports/eval_v2_full9350.log'
pid_file = ROOT / 'reports/eval_v2_full9350.pid'

cmd = [
    sys.executable,
    str(ROOT / 'scripts/eval_detector_full_9350_test.py'),
    '--gpu', '0',
    '--ckpt', 'artifacts/vrsbench_detector_v2_best.pth',
    '--arch', 'v2',
    '--mode', 'single'
]

with open(log_file, 'w') as f:
    proc = subprocess.Popen(
        cmd,
        stdout=f,
        stderr=subprocess.STDOUT,
        cwd=str(ROOT),
        start_new_session=True
    )

pid_file.write_text(str(proc.pid))
print(f"Launched Full 9350 test evaluation PID: {proc.pid}")
print(f"Logs: {log_file}")

