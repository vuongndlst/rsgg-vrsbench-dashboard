import subprocess, os, sys
from pathlib import Path

ROOT = Path.home() / 'rsg-former'

# Kill old overnight runner 360585 if alive
old_p = Path('/proc/360585')
if old_p.exists():
    print("Terminating old runner PID 360585...")
    subprocess.run(['kill', '360585'])
else:
    print("PID 360585 is not active.")

# Launch run_multi_dataset_overnight.py in background via nohup
log_path = ROOT / 'artifacts' / 'multi_dataset_overnight.log'
py_exe = ROOT / '.venv' / 'bin' / 'python'
script_path = ROOT / 'scripts' / 'run_multi_dataset_overnight.py'

cmd = f"nohup {py_exe} -u {script_path} > {log_path} 2>&1 & echo $!"
print("Launching command:", cmd)
proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
stdout, stderr = proc.communicate()
new_pid = stdout.strip()
print(f"NEW MASTER OVERNIGHT PIPELINE PID: {new_pid}")

# Save pid file
pid_file = ROOT / 'artifacts' / 'multi_dataset_overnight.pid'
pid_file.write_text(new_pid, encoding='utf-8')
print(f"Saved PID to {pid_file}")

