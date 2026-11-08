import os, glob
from pathlib import Path

hub_dir = str(Path.home() / ".cache/huggingface/hub")
found = []
for p in Path("/proc").glob("[0-9]*/fd/*"):
    try:
        target = os.readlink(str(p))
        if "huggingface" in target or "Qwen" in target:
            pid = p.parts[2]
            found.append((pid, str(p), target))
    except Exception:
        pass

print(f"Found {len(found)} matching open descriptors:")
for pid, fd, target in found[:20]:
    print(f"PID: {pid} -> {target}")

