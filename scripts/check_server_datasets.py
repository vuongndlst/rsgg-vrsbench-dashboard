from pathlib import Path

shared = Path.home() / 'shared-assets/datasets'
print("Datasets in shared-assets:")
if shared.exists():
    for p in shared.iterdir():
        if p.is_dir():
            print(f"  Dir: {p.name}/")
        else:
            print(f"  File: {p.name} ({p.stat().st_size / 1024**2:.1f} MB)")

sbir = Path.home() / 'sbir'
print("\nDatasets in ~/sbir:")
if sbir.exists():
    for p in sbir.iterdir():
        print(f"  {p.name}/")
else:
    print("  ~/sbir does not exist.")

