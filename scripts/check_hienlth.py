from pathlib import Path
p = Path.home() / 'sbir/hienlth'
if p.exists():
    for sub in p.iterdir():
        print(f"  {sub.name}")

