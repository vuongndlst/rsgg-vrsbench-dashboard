import subprocess
res = subprocess.run(['py-spy', 'dump', '--pid', '497614'], capture_output=True, text=True)
if res.returncode == 0:
    print(res.stdout)
else:
    print("py-spy err:", res.stderr)

