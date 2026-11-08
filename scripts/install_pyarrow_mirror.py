import sys, subprocess

mirror = "https://pypi.tuna.tsinghua.edu.cn/simple"
print(f"Installing pyarrow via mirror {mirror} using {sys.executable}...")
res = subprocess.run([
    sys.executable, '-m', 'pip', 'install', 'pyarrow',
    '-i', mirror, '--trusted-host', 'pypi.tuna.tsinghua.edu.cn'
], capture_output=True, text=True)

print("Returncode:", res.returncode)
print("Stdout:", res.stdout[-400:] if res.stdout else "")
print("Stderr:", res.stderr[-400:] if res.stderr else "")

import pyarrow
print("SUCCESS: pyarrow version", pyarrow.__version__)

