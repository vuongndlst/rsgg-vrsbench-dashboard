import zipfile, json
from pathlib import Path

SHARED = Path.home() / 'shared-assets/datasets/vrsbench'
ann_zip = SHARED / 'Annotations_train.zip'

with zipfile.ZipFile(ann_zip) as z:
    files = [f for f in z.namelist() if f.endswith('.json') and not f.startswith('__MACOSX')]
    print(f"Total annotation files: {len(files)}")
    boxes_count = 0
    non_empty = 0
    for f in files[:20]:
        d = json.loads(z.read(f).decode('utf-8'))
        objs = d.get('objects', [])
        print(f"File {f}: {len(objs)} objects")
        if objs:
            print("  Sample obj:", objs[0])
            non_empty += 1
            boxes_count += len(objs)
            
    print(f"\nIn first 20 files: {non_empty} files have objects, total objects: {boxes_count}")

