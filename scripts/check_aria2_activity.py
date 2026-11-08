import subprocess

print("=== ARIA2C NETWORK CONNECTIONS ===")
print(subprocess.getoutput("lsof -p 3625041 -i -a"))

print("\n=== ARIA2C FD INFO ===")
print(subprocess.getoutput("ls -l /proc/3625041/fd"))

print("\n=== ARIA2C STATS ===")
print(subprocess.getoutput("cat /proc/3625041/io"))

