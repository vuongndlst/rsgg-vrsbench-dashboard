import urllib.request, json

candidates = [
    "caglarmert/UCM-captions",
    "isaaccorley/UCM-Captions",
    "m-a-p/UCM-Captions",
    "flax-community/uc-merced-captions"
]

for c in candidates:
    try:
        url = f"https://huggingface.co/api/datasets/{c}"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as resp:
            meta = json.loads(resp.read().decode('utf-8'))
            print(f"FOUND: {c}, siblings: {len(meta.get('siblings', []))}")
            for s in meta.get('siblings', []):
                print(f"  file: {s.get('rfilename')}")
    except Exception as e:
        print(f"Not found or err {c}: {e}")

