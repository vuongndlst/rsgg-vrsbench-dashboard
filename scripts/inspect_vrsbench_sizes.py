import urllib.request, json

url = "https://huggingface.co/api/datasets/xiang709/VRSBench"
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
with urllib.request.urlopen(req, timeout=10) as resp:
    data = json.loads(resp.read().decode())
    for s in data.get('siblings', []):
        rfile = s.get('rfilename')
        # Check size with HEAD request
        f_url = f"https://huggingface.co/datasets/xiang709/VRSBench/resolve/main/{rfile}"
        try:
            h_req = urllib.request.Request(f_url, method='HEAD', headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(h_req, timeout=5) as h_resp:
                sz = int(h_resp.headers.get('Content-Length', 0))
                print(f"  {rfile}: {sz / (1024**2):.2f} MB")
        except Exception as e:
            print(f"  {rfile}: {e}")

