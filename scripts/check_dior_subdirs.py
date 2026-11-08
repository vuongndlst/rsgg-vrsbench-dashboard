from pathlib import Path
ROOT = Path.home() / 'rsg-former'
dior_dir = ROOT / 'data' / 'dior_rsvg'
all_files = list(dior_dir.rglob('*'))
print("Total rglob files in dior_dir:", len(all_files))
for p in all_files[:10]:
    print(" ", p.relative_to(dior_dir))

