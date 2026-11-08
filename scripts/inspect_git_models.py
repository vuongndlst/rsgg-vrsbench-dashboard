from pathlib import Path
for p in [Path.home() / 'models/git-large', Path.home() / 'models/git-large-coco']:
    print(f"=== {p} ===")
    if p.exists():
        for f in p.iterdir():
            print(f"  {f.name} ({f.stat().st_size / 1e6:.1f} MB)" if f.is_file() else f"  [DIR] {f.name}")
    else:
        print("  Not found")

