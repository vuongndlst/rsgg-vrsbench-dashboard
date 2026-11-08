import subprocess, os, sys
from pathlib import Path

# Check java
res = subprocess.run(['java', '-version'], capture_output=True, text=True)
print("Java returncode:", res.returncode)
print("Java stderr:", res.stderr.strip()[:200])

# Check pycocoevalcap spice location
try:
    import pycocoevalcap
    base = Path(pycocoevalcap.__file__).parent
    spice_dir = base / 'spice'
    print("SPICE dir:", spice_dir, "exists:", spice_dir.exists())
    if spice_dir.exists():
        for f in spice_dir.iterdir():
            print("  spice file:", f.name, f.stat().st_size)
        lib_dir = spice_dir / 'lib'
        if lib_dir.exists():
            print("  lib files:", [x.name for x in lib_dir.iterdir()])
except Exception as e:
    print("SPICE check err:", e)

