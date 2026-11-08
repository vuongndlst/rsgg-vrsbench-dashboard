import subprocess

out = subprocess.getoutput("ps aux | grep -E 'java|spice|python' | grep -v grep")
for l in out.splitlines():
    print(l[:120])

