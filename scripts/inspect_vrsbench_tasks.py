import json
from pathlib import Path

train_json = Path.home() / 'shared-assets/datasets/vrsbench/VRSBench_train.json'
with open(train_json) as f:
    data = json.load(f)

print(f"Total conversations: {len(data)}")

task_types = {}
for entry in data:
    conv = entry.get('conversations', [])
    if conv:
        prompt = conv[0].get('value', '')
        tag = prompt.split(']')[0].replace('<image>\n[', '').strip() if ']' in prompt else 'unknown'
        task_types[tag] = task_types.get(tag, 0) + 1

print("Task types in VRSBench_train:")
for k, v in sorted(task_types.items(), key=lambda x: -x[1]):
    print(f"  [{k}]: {v} pairs")

