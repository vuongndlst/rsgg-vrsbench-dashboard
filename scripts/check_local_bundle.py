import json
from pathlib import Path

p = Path("reports/dashboard_bundle.json")
data = json.loads(p.read_text(encoding="utf-8"))
print("Summary keys:", data["summary"].keys())
multi = data["summary"].get("multi_dataset_benchmarks", {})
print("Multi-dataset keys:", multi.keys())
for k in multi:
    print(f"  {k}: {list(multi[k].keys())}")

