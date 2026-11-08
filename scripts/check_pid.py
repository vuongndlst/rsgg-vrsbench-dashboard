import subprocess, os

pid = 497614
print(f"=== PID {pid} status ===")
print(subprocess.getoutput(f"ps -fp {pid}"))
print(subprocess.getoutput(f"cat /proc/{pid}/status | grep -E 'State|VmRSS|Threads'"))
print(subprocess.getoutput(f"ls -l /proc/{pid}/fd"))

