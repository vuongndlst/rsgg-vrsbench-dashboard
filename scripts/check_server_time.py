import os, time
from datetime import datetime

print("Current server time UTC:", datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'))
print("Current server time local:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

# Let's inspect the workers of PID 12173
p_dir = '/proc/12173/task'
if os.path.exists(p_dir):
    tasks = os.listdir(p_dir)
    print(f"PID 12173 thread count: {len(tasks)}")

# Let's find child processes of 12173
import subprocess
ps_out = subprocess.run(['ps', '-ef'], capture_output=True, text=True).stdout
for line in ps_out.splitlines():
    if '12173' in line or 'vrsbench' in line:
        print(line)

