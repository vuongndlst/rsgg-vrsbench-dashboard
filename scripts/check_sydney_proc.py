import subprocess, os
res = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
for line in res.stdout.splitlines():
    if 'inspect_sydney' in line or 'python' in line:
        print(line[:120])

