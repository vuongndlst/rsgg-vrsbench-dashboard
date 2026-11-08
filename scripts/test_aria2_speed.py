import subprocess, time, os
from pathlib import Path

test_file = Path.home() / 'rsg-former/scratch/test_dl.zip'
test_file.parent.mkdir(parents=True, exist_ok=True)
if test_file.exists(): test_file.unlink()

url = "http://images.cocodataset.org/zips/train2014.zip"
cmd = f"aria2c -x 16 -s 16 -k 1M --stop=10 -d {test_file.parent} -o {test_file.name} {url}"

t0 = time.time()
out = subprocess.getoutput(cmd)
elapsed = time.time() - t0
size = test_file.stat().st_size if test_file.exists() else 0
if test_file.exists(): test_file.unlink()

print(f"Downloaded {size / 1e6:.1f} MB in {elapsed:.2f}s -> Speed: {(size / 1e6) / max(elapsed, 0.01):.2f} MB/s")
print("Aria2 output preview:")
for line in out.splitlines()[-6:]:
    print(" ", line)

