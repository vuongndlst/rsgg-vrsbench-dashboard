import subprocess
out = subprocess.check_output(["nvidia-smi"], text=True)
print(out)

