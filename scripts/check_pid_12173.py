import os
from pathlib import Path

pid = 12173
print(f"=== CHECK PID {pid} ===")
p = Path(f"/proc/{pid}")
if p.exists():
    print("Cmdline:", p.joinpath("cmdline").read_bytes().replace(b'\x00', b' ').decode('utf-8', errors='ignore'))
    print("Stat:", p.joinpath("stat").read_text().split()[:15])
    fd_dir = p.joinpath("fd")
    if fd_dir.exists():
        for fd in fd_dir.iterdir():
            try:
                print(f"fd {fd.name} -> {os.readlink(str(fd))}")
            except Exception:
                pass

# Also check artifacts/reports
for f in Path.home().joinpath("rsg-former/reports").glob("*"):
    print("Report file:", f.name, f.stat().st_size, "bytes", f.stat().st_mtime)

for f in Path.home().joinpath("rsg-former/artifacts").glob("vrsbench_detector_v2*"):
    print("Checkpoint:", f.name, f.stat().st_size, "bytes", f.stat().st_mtime)

