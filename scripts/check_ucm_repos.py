import urllib.request, json

for repo in ["cpratikaki/UCMcaptions_finetuning", "caglarmert/UCM-captions"]:
    url = f"https://huggingface.co/api/datasets/{repo}"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'}), timeout=10) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            print("REPO:", repo)
            for s in data.get('siblings', []):
                print("  file:", s.get('rfilename'))
    except Exception as e:
        print("ERR:", repo, e)

