from pathlib import Path
import json

p = Path.home() / 'rsg-former/reports/COCO_HYBRID_RSG_GIT_PILOT100_RESULT.json'
print(p.read_text())

