from pathlib import Path
import time

for pid in [3922367, 3922819, 3927854, 3927855, 3927856, 3927857]:
    p = Path(f"/proc/{pid}/io")
    s = Path(f"/proc/{pid}/stat")
    if p.exists():
        print(f"PID {pid} IO:\n{p.read_text().strip()}")
    if s.exists():
        parts = s.read_text().split()
        print(f"PID {pid} utime={parts[13]} stime={parts[14]}")
