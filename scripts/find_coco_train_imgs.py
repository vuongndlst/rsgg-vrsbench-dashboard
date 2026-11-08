import glob
from pathlib import Path
for pattern in ['/home/stackops/shared-assets/**/train2014*', '/home/stackops/**/coco*']:
    matches = glob.glob(pattern, recursive=True)
    if matches:
        print(f"Pattern {pattern}: {len(matches)} matches")
        for m in matches[:5]:
            print(f"  {m}")

