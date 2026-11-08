from pathlib import Path
dior_dir = Path.home() / 'rsg-former' / 'data' / 'dior_rsvg' / 'Annotations'
xmls = sorted(list(dior_dir.glob("*.xml")))
print(f"Total: {len(xmls)}")
if xmls:
    print(f"First xml name: {xmls[0].name}")
    print(xmls[0].read_text(encoding='utf-8')[:800])
