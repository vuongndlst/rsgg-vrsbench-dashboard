import os, subprocess
pid = 542016
out = subprocess.getoutput(f"ls -l /proc/{pid}/fd")
print(out)
