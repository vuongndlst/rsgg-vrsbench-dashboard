import subprocess

out = subprocess.getoutput("ps -eo pid,user,etime,%cpu,%mem,args | grep -E 'python' | grep -v grep")
for l in out.splitlines():
    if any(k in l for k in ["sydney", "rsg-former", "eval"]):
        print(l[:120])
