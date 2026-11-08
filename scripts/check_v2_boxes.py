import json
from pathlib import Path

p = Path.home() / 'rsg-former' / 'reports' / 'VRSBENCH_DETECTOR_V2_SINGLE_BOXES_9350.json'
with open(p, 'r') as f:
    data = json.load(f)
dets = data['detections']
fk = list(dets.keys())[0]
print(f"Sample key: {fk}, type: {type(dets[fk])}")
if isinstance(dets[fk], dict):
    print("Keys:", dets[fk].keys())
    for subk, subv in list(dets[fk].items())[:4]:
        print(f"  {subk}: {subv}")
