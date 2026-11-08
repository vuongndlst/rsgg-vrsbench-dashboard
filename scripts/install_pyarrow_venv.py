import sys, subprocess

print("Installing pyarrow via:", sys.executable)
res = subprocess.run([sys.executable, '-m', 'pip', 'install', 'pyarrow', '--no-cache-dir'], capture_output=True, text=True)
print("Returncode:", res.returncode)
print("Stdout tail:", res.stdout[-300:] if res.stdout else "")
print("Stderr tail:", res.stderr[-300:] if res.stderr else "")

import pyarrow
print("Successfully imported pyarrow version:", pyarrow.__version__)

