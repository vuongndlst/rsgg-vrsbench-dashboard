import subprocess
res = subprocess.run(['ps', '-u', 'stackops', '-o', 'pid,%cpu,%mem,stat,etime,command'], capture_output=True, text=True)
for line in res.stdout.splitlines():
    if 'python' in line:
        print(line[:120])

