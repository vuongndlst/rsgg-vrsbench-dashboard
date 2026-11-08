from pathlib import Path
p = Path.home() / 'shared-assets/datasets/rsicd'
if p.exists():
    for f in p.iterdir():
        print(f"  {f.name}: {f.stat().st_size / (1024**2):.2f} MB")
else:
    print("Does not exist yet.")

