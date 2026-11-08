import json
from pathlib import Path

shared = Path.home() / 'shared-assets/datasets/vrsbench'
p = shared / 'VRSBench_EVAL_Cap.json'
if p.exists():
    with open(p) as f:
        d = json.load(f)
    print("VRSBench_EVAL_Cap.json total items:", len(d))
    print("Sample 0:", d[0])
else:
    print("File not found:", p)

