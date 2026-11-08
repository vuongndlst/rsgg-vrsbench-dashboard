from pathlib import Path
home = Path.home()
print("Checking home model candidates:")
for p in [home / 'models', home / 'shared-assets' / 'models', home / '.cache' / 'huggingface' / 'hub']:
    if p.exists():
        print(f"Path {p}:")
        for sub in p.iterdir():
            print(" ", sub.name)

