import urllib.request

url = "https://huggingface.co/datasets/arampacha/rsicd/resolve/main/data/test-00000-of-00001.parquet"
req = urllib.request.Request(url, method='HEAD', headers={'User-Agent': 'Mozilla/5.0'})
try:
    with urllib.request.urlopen(req, timeout=10) as resp:
        length = int(resp.headers.get('Content-Length', 0))
        print(f"test parquet size: {length / (1024*1024):.2f} MB")
except Exception as e:
    print("Error:", e)

