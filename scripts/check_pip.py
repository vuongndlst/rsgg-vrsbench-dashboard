import subprocess
res = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
for line in res.stdout.splitlines():
    if 'pip' in line or 'install_pyarrow' in line:
        print(line[:120])

