import subprocess

for tool in ['aria2c', 'axel', 'wget', 'curl', 'unzip']:
    path = subprocess.getoutput(f"which {tool}")
    print(f"Tool {tool}: {path if path else 'NOT FOUND'}")

