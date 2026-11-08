import urllib.request, json

candidates = [
    "tianlinlin/UCM-Captions",
    "arwhughes/ucm-captions",
    "gokul-pv/ucm-captions",
    "BigEarthNet/ucm",
    "UCMerced/LandUse"
]
for c in candidates:
    try:
        url = f"https://huggingface.co/api/datasets/{c}"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as resp:
            meta = json.loads(resp.read().decode('utf-8'))
            print(f"FOUND: {c}, siblings: {len(meta.get('siblings', []))}")
    except:
        pass

# Check pyarrow and pandas
try:
    import pandas as pd
    print("pandas installed:", pd.__version__)
except ImportError:
    print("pandas NOT installed")

try:
    import pyarrow as pa
    print("pyarrow installed:", pa.__version__)
except ImportError:
    print("pyarrow NOT installed")

