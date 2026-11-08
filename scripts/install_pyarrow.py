import subprocess

res = subprocess.run(['pip', 'install', 'pyarrow', 'datasets'], capture_output=True, text=True)
print("pip install rc:", res.returncode)
print("pip install output:", res.stdout[-300:] if res.stdout else res.stderr[-300:])

import pyarrow
print("pyarrow version:", pyarrow.__version__)
import datasets
print("datasets version:", datasets.__version__)

