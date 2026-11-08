from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path.home() / 'rsg-former'
xml_file = ROOT / 'data' / 'dior_rsvg' / 'Annotations' / '21195.xml'
tree = ET.parse(xml_file)
root = tree.getroot()
print("Root tag:", root.tag)
for elem in root:
    print(f"  {elem.tag}: {elem.text.strip() if elem.text else [c.tag for c in elem]}")
    if elem.tag == 'object':
        for c in elem:
            if len(c):
                print(f"    {c.tag}: {[x.tag + '=' + str(x.text) for x in c]}")
            else:
                print(f"    {c.tag}: {c.text}")

