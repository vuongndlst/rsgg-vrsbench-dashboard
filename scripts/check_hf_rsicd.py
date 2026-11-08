import urllib.request, json

url = "https://huggingface.co/api/datasets/arampacha/rsicd"
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
try:
    res = urllib.request.urlopen(req, timeout=10).read().decode('utf-8')
    data = json.loads(res)
    siblings = [s['rfilename'] for s in data.get('siblings', [])]
    print("Files in arampacha/rsicd:", siblings)
except Exception as e:
    print("HF API error:", e)

