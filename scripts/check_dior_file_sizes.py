import urllib.request, json

url = "https://huggingface.co/api/datasets/LittleCollections/DIOR-RSVG"
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
with urllib.request.urlopen(req) as resp:
    data = json.loads(resp.read().decode('utf-8'))
    for s in data.get('siblings', []):
        fn = s.get('rfilename')
        # fetch headers with HEAD request to get Content-Length
        head_url = f"https://huggingface.co/datasets/LittleCollections/DIOR-RSVG/resolve/main/{fn}"
        h_req = urllib.request.Request(head_url, method='HEAD', headers={'User-Agent': 'Mozilla/5.0'})
        try:
            with urllib.request.urlopen(h_req) as h_resp:
                sz = int(h_resp.headers.get('Content-Length', 0))
                print(f"  {fn}: {sz / (1024*1024):.2f} MB")
        except Exception as e:
            print(f"  {fn}: size unknown ({e})")

