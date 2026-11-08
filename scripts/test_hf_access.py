import urllib.request, json

# Check huggingface_hub
try:
    import huggingface_hub
    print("huggingface_hub version:", huggingface_hub.__version__)
except ImportError:
    print("huggingface_hub NOT installed")

# Test downloading small metadata or raw file from HF
try:
    url = "https://huggingface.co/api/datasets/isaaccorley/Sydney-Captions"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=10) as resp:
        meta = json.loads(resp.read().decode('utf-8'))
        print("HF API Sydney-Captions accessible:", meta.get('id'), "siblings:", len(meta.get('siblings', [])))
        for s in meta.get('siblings', [])[:5]:
            print("  file:", s.get('rfilename'))
except Exception as e:
    print("HF API test error:", e)

