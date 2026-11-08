import os
from pathlib import Path

cand = Path.home() / "etechs/msvd-tkg-captioning/models"
print("Candidate path:", cand)
if cand.exists():
    for p in cand.glob("*"):
        print(p.name, p.is_dir())
        if p.is_dir():
            files = list(p.glob("*"))
            print(f"   contains {len(files)} files, e.g. {[f.name for f in files[:5]]}")
else:
    print("Does not exist.")

# Also check other shared models
for p in Path.home().glob("**/models/*"):
    if p.is_dir() and "qwen" in p.name.lower():
        print("Found:", p)

