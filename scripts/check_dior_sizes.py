import urllib.request, json

url = "https://huggingface.co/api/datasets/danielz01/DIOR-RSVG"
try:
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=10) as resp:
        meta = json.loads(resp.read().decode('utf-8'))
        print("danielz01/DIOR-RSVG files:")
        for s in meta.get('siblings', []):
            if 'test' in s.get('rfilename'):
                print(" ", s.get('rfilename'))
except Exception as e:
    print("Err danielz01:", e)

url2 = "https://huggingface.co/api/datasets/LittleCollections/DIOR-RSVG"
try:
    req = urllib.request.Request(url2, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=10) as resp:
        meta = json.loads(resp.read().decode('utf-8'))
        print("LittleCollections/DIOR-RSVG files:")
        for s in meta.get('siblings', []):
            print(" ", s.get('rfilename'))
except Exception as e:
    print("Err LittleCollections:", e)

