import os, glob

target_pipe = "7271915"
print(f"Searching for reader of pipe:{target_pipe}...")

found = []
for p in glob.glob('/proc/[0-9]*/fd/*'):
    try:
        link = os.readlink(p)
        if target_pipe in link:
            pid = p.split('/')[2]
            fd = p.split('/')[4]
            found.append((pid, fd, link))
    except Exception:
        pass

for pid, fd, link in found:
    cmd = "unknown"
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            cmd = f.read().replace(b'\x00', b' ').decode('utf-8', errors='ignore')
    except Exception:
        pass
    print(f"PID {pid} fd {fd} -> {link} | CMD: {cmd[:100]}")

