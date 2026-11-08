import urllib.request, time

mirrors = [
    "https://huggingface.co/datasets/HuggingFaceM4/COCO/resolve/main/data/train2014.zip",
    "https://hf-mirror.com/datasets/HuggingFaceM4/COCO/resolve/main/data/train2014.zip",
    "http://images.cocodataset.org/zips/train2014.zip"
]

for url in mirrors:
    try:
        t0 = time.time()
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = resp.read(5 * 1024 * 1024) # 5MB
            elapsed = time.time() - t0
            speed = (len(data) / 1e6) / max(elapsed, 0.01)
            print(f"URL: {url[:60]}... -> OK (status {resp.status}, speed: {speed:.2f} MB/s)")
    except Exception as e:
        print(f"URL: {url[:60]}... -> Failed ({e})")

