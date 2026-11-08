import subprocess

out = subprocess.getoutput("ps -eo pid,user,etime,%cpu,%mem,args | grep -E 'python -B -' | grep -v grep")
print(out)
