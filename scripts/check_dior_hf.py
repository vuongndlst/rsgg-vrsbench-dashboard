import urllib.request, json

candidates = [
    "danielz01/DIOR-RSVG",
    "LittleCollections/DIOR-RSVG",
    "pzhang1990/DIOR-RSVG"
]

for repo in candidates:
    url = f"https://huggingface.co/api/datasets/{repo}"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            print("FOUND DIOR REPO:", repo)
            for s in data.get('siblings', [])[:15]:
                print("  file:", s.get('rfilename'))
    except Exception as e:
        print("ERR:", repo, e)

