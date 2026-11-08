import urllib.request, json

url = "https://huggingface.co/api/datasets/arampacha/rsicd"
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
with urllib.request.urlopen(req, timeout=10) as resp:
    data = json.loads(resp.read().decode())
    for s in data.get('siblings', []):
        print(f"  {s.get('rfilename')}")

