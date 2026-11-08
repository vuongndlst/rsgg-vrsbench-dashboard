import urllib.request, json

url = "https://huggingface.co/api/datasets/xiang709/VRSBench"
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
with urllib.request.urlopen(req, timeout=10) as resp:
    data = json.loads(resp.read().decode())
    siblings = data.get('siblings', [])
    print(f"Total files in xiang709/VRSBench: {len(siblings)}")
    for s in siblings[:20]:
        print(f"  {s.get('rfilename')}")

