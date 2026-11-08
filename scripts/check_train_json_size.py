from pathlib import Path
p = Path.home() / 'shared-assets/datasets/vrsbench/VRSBench_train.json'
if p.exists():
    print(f"Current size: {p.stat().st_size / 1024**2:.2f} MB")
else:
    print("Does not exist yet.")

