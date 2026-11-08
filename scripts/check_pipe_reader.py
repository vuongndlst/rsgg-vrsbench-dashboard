import subprocess

print("=== CHECK PIPE 3463222 ===")
print(subprocess.getoutput("grep -l '3463222' /proc/*/fd/* 2>/dev/null"))

print("\n=== CHECK PARENT OF 3625040 & 3625041 ===")
print(subprocess.getoutput("ps -f -p 3625040,3625041"))
print(subprocess.getoutput("cat /proc/3625040/status | grep PPid"))

