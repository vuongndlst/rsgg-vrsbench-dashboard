import urllib.request, json
from pathlib import Path

target_dir = Path.home() / 'shared-assets/datasets/vrsbench'
target_dir.mkdir(parents=True, exist_ok=True)
eval_json = target_dir / 'VRSBench_EVAL_Cap.json'

url = "https://huggingface.co/datasets/xiang709/VRSBench/resolve/main/VRSBench_EVAL_Cap.json"
print(f"Downloading {url} to {eval_json}...")
urllib.request.urlretrieve(url, eval_json)
print(f"Downloaded! Size: {eval_json.stat().st_size / (1024**2):.2f} MB")

with open(eval_json) as f:
    data = json.load(f)

print(f"Total entries in EVAL_Cap: {len(data)}")
sample = list(data.items())[0] if isinstance(data, dict) else data[0]
print("Sample entry:\n", json.dumps(sample, indent=2))

