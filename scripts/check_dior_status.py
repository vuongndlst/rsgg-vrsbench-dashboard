from pathlib import Path
dior_dir = Path.home() / 'rsg-former' / 'data' / 'dior_rsvg'
print("DIOR-RSVG dir exists:", dior_dir.exists())
test_file = dior_dir / 'test.txt'
if test_file.exists():
    lines = test_file.read_text(encoding='utf-8').splitlines()
    print(f"Total lines in test.txt: {len(lines)}")
    print(f"First 5 lines: {lines[:5]}")
ann_dir = dior_dir / 'Annotations'
if ann_dir.exists():
    xmls = list(ann_dir.glob("*.xml"))
    print(f"Total XMLs in Annotations: {len(xmls)}")

