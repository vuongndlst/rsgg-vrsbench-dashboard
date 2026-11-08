import urllib.request, json

# Search Hugging Face dataset API for 'ucm' or 'uc-merced'
url = "https://huggingface.co/api/datasets?search=ucm-captions"
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
try:
    with urllib.request.urlopen(req, timeout=10) as resp:
        results = json.loads(resp.read().decode('utf-8'))
        print(f"Found {len(results)} datasets matching 'ucm-captions':")
        for r in results[:10]:
            print(" ", r.get('id'))
except Exception as e:
    print("HF search error:", e)

# Also search 'uc merced captions'
url2 = "https://huggingface.co/api/datasets?search=uc+merced"
try:
    with urllib.request.urlopen(urllib.request.Request(url2, headers={'User-Agent': 'Mozilla/5.0'}), timeout=10) as resp:
        results2 = json.loads(resp.read().decode('utf-8'))
        print(f"Found {len(results2)} datasets matching 'uc merced':")
        for r in results2[:10]:
            print(" ", r.get('id'))
except Exception as e:
    print("HF search 2 error:", e)

