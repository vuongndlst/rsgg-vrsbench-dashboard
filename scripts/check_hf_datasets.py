import urllib.request, json

urls = [
    "https://huggingface.co/api/datasets/arampacha/rsicd",
    "https://huggingface.co/api/datasets/xiang709/VRSBench"
]

for url in urls:
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            print(f"URL: {url} -> OK!")
            print(f"  ID: {data.get('id')}")
            print(f"  Author: {data.get('author')}")
            print(f"  Downloads: {data.get('downloads')}")
    except Exception as e:
        print(f"URL: {url} -> Failed: {e}")

