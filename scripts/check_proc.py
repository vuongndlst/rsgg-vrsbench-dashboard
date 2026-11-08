import subprocess

out = subprocess.run(["ps", "-eo", "pid,pcpu,pmem,etime,cmd"], capture_output=True, text=True).stdout
for line in out.splitlines():
    if "python" in line and not "check_proc" in line:
        print(line)
