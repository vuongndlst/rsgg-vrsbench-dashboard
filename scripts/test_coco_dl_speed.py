import time, urllib.request

url = "http://images.cocodataset.org/zips/train2014.zip"
t0 = time.time()
bytes_read = 0
chunk_size = 1024 * 1024  # 1 MB

print("Testing download speed from images.cocodataset.org...")
with urllib.request.urlopen(url, timeout=10) as resp:
    while bytes_read < 20 * 1024 * 1024:  # read 20 MB
        chunk = resp.read(chunk_size)
        if not chunk: break
        bytes_read += len(chunk)

elapsed = time.time() - t0
speed_mb = (bytes_read / (1024 * 1024)) / elapsed
print(f"Downloaded {bytes_read / 1e6:.1f} MB in {elapsed:.2f}s -> Speed: {speed_mb:.2f} MB/s")
eta_min = (12.58 * 1024 / speed_mb) / 60
print(f"Estimated time for full 12.58 GB train2014.zip: {eta_min:.1f} minutes")

