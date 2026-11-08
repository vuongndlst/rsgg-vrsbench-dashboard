import subprocess

res = subprocess.run(['sudo', 'apt-get', 'install', '-y', 'python3-pyarrow'], capture_output=True, text=True)
print("apt install python3-pyarrow rc:", res.returncode)
print("apt stdout tail:", res.stdout[-200:] if res.stdout else "")

try:
    import pyarrow
    print("pyarrow imported in system python:", pyarrow.__version__)
except Exception as e:
    print("import err:", e)

