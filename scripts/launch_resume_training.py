import subprocess, os, sys
from pathlib import Path

ROOT = Path.home() / 'rsg-former'
log_file = ROOT / 'artifacts/vrsbench_detector_v2_train.log'
pid_file = ROOT / 'artifacts/vrsbench_detector_v2_train.pid'

# Launch detached process
cmd = [sys.executable, str(ROOT / 'scripts/resume_train_vrsbench_detector_v2.py')]
with open(log_file, 'a') as f:
    proc = subprocess.Popen(
        cmd,
        stdout=f,
        stderr=subprocess.STDOUT,
        cwd=str(ROOT),
        start_new_session=True
    )

pid_file.write_text(str(proc.pid))
print(f"Launched resume training process PID: {proc.pid}")
print(f"Logs: {log_file}")

