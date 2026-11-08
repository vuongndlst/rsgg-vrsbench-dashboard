from pathlib import Path
import json

root = Path.home() / 'rsg-former'
artifacts = root / 'artifacts'
print("Artifacts directory files:")
for p in artifacts.iterdir():
    if p.is_file():
        print(f"  {p.name} ({p.stat().st_size / 1024:.1f} KB)")
    elif p.is_dir():
        print(f"  {p.name}/")

reports = root / 'reports'
if reports.exists():
    print("\nReports directory files:")
    for p in reports.iterdir():
        print(f"  {p.name} ({p.stat().st_size / 1024:.1f} KB)")

