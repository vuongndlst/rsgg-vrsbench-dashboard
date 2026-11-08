import urllib.request, time

urls = [
    "https://pypi.org",
    "https://pypi.tuna.tsinghua.edu.cn",
    "https://mirrors.aliyun.com/pypi/simple/",
    "https://huggingface.co"
]
for u in urls:
    t0 = time.time()
    try:
        req = urllib.request.Request(u, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as resp:
            print(f"OK ({time.time()-t0:.2f}s): {u} -> status {resp.status}")
    except Exception as e:
        print(f"FAIL ({time.time()-t0:.2f}s): {u} -> {e}")

