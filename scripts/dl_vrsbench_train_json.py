import urllib.request, json
from pathlib import Path

target_dir = Path.home() / 'shared-assets/datasets/vrsbench'
target_dir.mkdir(parents=True, exist_ok=True)
train_json = target_dir / 'VRSBench_train.json'

if not train_json.exists():
    url = "https://huggingface.co/datasets/xiang709/VRSBench/resolve/main/VRSBench_train.json"
    print(f"Downloading {url} to {train_json}...")
    urllib.request.urlretrieve(url, train_json)
    print(f"Downloaded! Size: {train_json.stat().st_size / (1024**2):.2f} MB")
else:
    print(f"Already exists! Size: {train_json.stat().st_size / (1024**2):.2f} MB")

with open(train_json) as f:
    data = json.load(f)

print(f"Total entries in VRSBench_train: {len(data)}")
sample = list(data.items())[0] if isinstance(data, dict) else data[0]
print("Sample train entry keys:", list(sample.keys()) if isinstance(sample, dict) else type(sample))
print("Sample train entry preview:\n", json.dumps(sample, indent=2)[:500])

