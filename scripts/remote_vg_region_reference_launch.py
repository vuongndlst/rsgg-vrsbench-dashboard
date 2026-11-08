"""Launch one independent, bounded CPU metadata preparation; never starts training."""
from pathlib import Path
import json
import os
import subprocess
import time

root = Path.home() / 'rsg-former'
dest = root / 'artifacts/caption_readiness'
dest.mkdir(parents=True, exist_ok=True)
runner = root / 'scripts/remote_vg_region_reference_prepare.py'
pidfile = dest / 'region_prepare.pid'
if pidfile.exists():
    pid = int(pidfile.read_text())
    cmdline = Path('/proc') / str(pid) / 'cmdline'
    if cmdline.exists() and str(runner).encode() in cmdline.read_bytes():
        print(json.dumps({'status': 'already_running', 'pid': pid})); raise SystemExit
env = dict(os.environ)
env.update(PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
with (dest / 'region_prepare.log').open('ab', buffering=0) as log:
    proc = subprocess.Popen(['/usr/bin/python3', '-B', str(runner)], cwd=root, env=env,
                            stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                            start_new_session=True)
pidfile.write_text(str(proc.pid))
time.sleep(1)
print(json.dumps({'status': 'started' if proc.poll() is None else 'early_exit', 'pid': proc.pid,
                  'report': 'reports/VG_REGION_REFERENCE_PREPARATION.json'}))
