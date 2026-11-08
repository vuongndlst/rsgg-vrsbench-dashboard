import time, subprocess

def get_io():
    try:
        with open("/proc/3625041/io") as f:
            for line in f:
                if line.startswith("write_bytes:"):
                    return int(line.split()[1])
    except Exception as e:
        return str(e)
    return None

w1 = get_io()
time.sleep(2)
w2 = get_io()

print(f"w1: {w1}, w2: {w2}, delta: {w2 - w1 if isinstance(w1, int) and isinstance(w2, int) else 'N/A'}")

