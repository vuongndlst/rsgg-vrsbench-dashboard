import os
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path.home() / 'rsg-former'
dior_dir = ROOT / 'data' / 'dior_rsvg'
xmls = list(dior_dir.glob('*.xml'))
print("Total xml files:", len(xmls))
if xmls:
    print("Reading sample xml:", xmls[0].name)
    tree = ET.parse(xmls[0])
    root = tree.getroot()
    print("Root tag:", root.tag)
    for elem in root:
        print(f"  {elem.tag}: {elem.text if not len(elem) else [c.tag for c in elem]}")
        if elem.tag == 'object':
            for c in elem:
                print(f"    {c.tag}: {c.text}")

