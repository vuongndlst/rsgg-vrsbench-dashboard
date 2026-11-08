import subprocess, os, sys
from pathlib import Path

# Check if java exists anywhere in /usr or /opt
res = subprocess.run(['find', '/usr', '/opt', '-name', 'java', '-type', 'f'], capture_output=True, text=True)
print("Java candidates in /usr, /opt:")
print(res.stdout[:500])

# Check sudo
res_sudo = subprocess.run(['sudo', '-n', 'true'], capture_output=True, text=True)
print("Sudo without password:", res_sudo.returncode == 0)

# Check pycocoevalcap
try:
    import pycocoevalcap
    base = Path(pycocoevalcap.__file__).parent
    spice_dir = base / 'spice'
    print("SPICE dir:", spice_dir, "exists:", spice_dir.exists())
    if spice_dir.exists():
        for f in spice_dir.iterdir():
            print("  spice file:", f.name, f.stat().st_size)
except Exception as e:
    print("SPICE check err:", e)

