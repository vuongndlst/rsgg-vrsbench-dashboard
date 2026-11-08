"""Launch bounded CPU-only metadata preparation without touching v2 training."""
from pathlib import Path
import json
import os
import subprocess
import time

root = Path.home() / 'rsg-former'
folder = root / 'artifacts/caption_readiness'
folder.mkdir(parents=True, exist_ok=True)
runner = root / 'scripts/remote_vg_canonical_annotation_prepare.py'
pidfile = folder / 'canonical_annotation_prepare.pid'
report = root / 'reports/VG_CANONICAL_ANNOTATION_PREPARATION.json'
if report.exists() and json.loads(report.read_text()).get('status') == 'download_complete_parity_pending':
    print(json.dumps({'status': 'already_prepared'})); raise SystemExit
if pidfile.exists():
    pid = int(pidfile.read_text())
    cmd = Path('/proc') / str(pid) / 'cmdline'
    if cmd.exists() and str(runner).encode() in cmd.read_bytes():
        print(json.dumps({'status': 'already_running', 'pid': pid})); raise SystemExit
env = dict(os.environ)
env.update(PYTHONPATH=str(root / '.tools/deps'), PYTHONDONTWRITEBYTECODE='1',
           OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
with (folder / 'canonical_annotation_prepare.log').open('ab', buffering=0) as log:
    proc = subprocess.Popen([str(root / '.venv/bin/python'), '-B', str(runner)], cwd=root, env=env,
                            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)
pidfile.write_text(str(proc.pid))
time.sleep(1)
print(json.dumps({'status': 'started' if proc.poll() is None else 'early_exit', 'pid': proc.pid,
                  'report': 'reports/VG_CANONICAL_ANNOTATION_PREPARATION.json'}))
