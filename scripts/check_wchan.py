import subprocess

pid = 497614
print("wchan:", subprocess.getoutput(f"cat /proc/{pid}/wchan"))
print("stack:\n", subprocess.getoutput(f"cat /proc/{pid}/stack"))

