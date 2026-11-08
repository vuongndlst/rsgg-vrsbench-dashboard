import subprocess
pid = 542016
print(subprocess.getoutput(f"ps -fp {pid}"))
print(subprocess.getoutput(f"cat /proc/{pid}/status | grep -E 'State|VmRSS|Threads'"))

