import subprocess, sys

# Kill any leftover hanging pip
subprocess.run(['pkill', '-f', 'pip'])

print("Installing pyarrow directly from pypi.org...")
res = subprocess.run([sys.executable, '-m', 'pip', 'install', 'pyarrow', '--index-url', 'https://pypi.org/simple'], capture_output=True, text=True)
print("Returncode:", res.returncode)
print("Stdout:", res.stdout[-400:] if res.stdout else "")
print("Stderr:", res.stderr[-400:] if res.stderr else "")

import pyarrow
print("SUCCESS: pyarrow version:", pyarrow.__version__)

