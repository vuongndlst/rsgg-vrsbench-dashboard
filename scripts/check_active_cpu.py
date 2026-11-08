import time

def get_cpu():
    with open('/proc/12173/stat') as f:
        parts = f.read().split()
        return int(parts[13]) + int(parts[14])

c1 = get_cpu()
time.sleep(2)
c2 = get_cpu()
diff = c2 - c1
print(f"CPU jiffies diff in 2 seconds: {diff} (active rate: {diff/2} jiffies/s, ~{diff/2}% of 1 CPU core)")

