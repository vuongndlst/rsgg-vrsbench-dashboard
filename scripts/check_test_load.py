import subprocess
res = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
for line in res.stdout.splitlines():
    if 'test_engine_load' in line:
        print(line[:120])

