import subprocess, urllib.request

print("=== DISK SPACE (df -h) ===")
print(subprocess.getoutput("df -h /home/stackops"))

print("\n=== COCO OFFICIAL ZIP URL HEAD CHECK ===")
url = "http://images.cocodataset.org/zips/train2014.zip"
try:
    req = urllib.request.Request(url, method='HEAD')
    with urllib.request.urlopen(req, timeout=10) as resp:
        size_gb = int(resp.headers.get('Content-Length', 0)) / (1024**3)
        print(f"URL: {url}")
        print(f"Status: {resp.status}")
        print(f"Size: {size_gb:.2f} GB")
except Exception as e:
    print(f"Error checking {url}: {e}")

