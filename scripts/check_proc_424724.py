import subprocess
res = subprocess.run(['ps', '-fp', '424724'], capture_output=True, text=True)
print(res.stdout)

