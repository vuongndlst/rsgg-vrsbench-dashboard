import json
from pathlib import Path

p = Path.home() / 'rsg-former' / 'reports' / 'VRSBENCH_DETECTOR_V2_SG_FULL9350_CACHE.json'
if p.exists():
    with open(p) as f:
        d = json.load(f)
    print(f"Total SG entries: {len(d)}")
    for k in list(d.keys())[:5]:
        print(f"QID {k}: {d[k]}")
else:
    print("Not found")

