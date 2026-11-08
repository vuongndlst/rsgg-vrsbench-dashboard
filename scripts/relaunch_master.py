import subprocess
from pathlib import Path

ROOT = Path.home() / 'rsg-former'

# Kill old processes
for pid in [489034, 436958]:
    p = Path(f'/proc/{pid}')
    if p.exists():
        print(f"Terminating PID {pid}...")
        subprocess.run(['kill', '-9', str(pid)])

# Also kill any python process running eval_sydney or auto_tune
subprocess.run(['pkill', '-9', '-f', 'eval_sydney_captions.py'])
subprocess.run(['pkill', '-9', '-f', 'run_multi_dataset_overnight.py'])

# Re-launch run_multi_dataset_overnight.py
log_path = ROOT / 'artifacts' / 'multi_dataset_overnight.log'
py_exe = ROOT / '.venv' / 'bin' / 'python'
script_path = ROOT / 'scripts' / 'run_multi_dataset_overnight.py'

cmd = f"nohup {py_exe} -u {script_path} > {log_path} 2>&1 & echo $!"
print("Launching command:", cmd)
proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
stdout, stderr = proc.communicate()
new_pid = stdout.strip()
print(f"RELAUNCHED MASTER OVERNIGHT PIPELINE PID: {new_pid}")

pid_file = ROOT / 'artifacts' / 'multi_dataset_overnight.pid'
pid_file.write_text(new_pid, encoding='utf-8')

