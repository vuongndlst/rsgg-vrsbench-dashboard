import subprocess, sys
from pathlib import Path

ROOT = Path.home() / 'rsg-former'
log_file = ROOT / 'artifacts/overnight_pipeline.log'
pid_file = ROOT / 'artifacts/overnight_pipeline.pid'

cmd = [sys.executable, str(ROOT / 'scripts/run_vrsbench_overnight_pipeline.py')]
with open(log_file, 'a') as f:
    proc = subprocess.Popen(
        cmd,
        stdout=f,
        stderr=subprocess.STDOUT,
        cwd=str(ROOT),
        start_new_session=True
    )

pid_file.write_text(str(proc.pid))
print(f"Launched VRSBench Master Overnight Pipeline PID: {proc.pid}")
print(f"Log file: {log_file}")

