import subprocess
ps = subprocess.getoutput("ps aux | grep aria2 | grep -v grep")
print("Aria2 processes:")
print(ps)

