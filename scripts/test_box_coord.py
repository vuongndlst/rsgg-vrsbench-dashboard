import zipfile, json
from pathlib import Path

SHARED = Path.home() / 'shared-assets/datasets/vrsbench'
ann_zip = SHARED / 'Annotations_train.zip'

with zipfile.ZipFile(ann_zip) as z:
    files = [f for f in z.namelist() if f.endswith('.json') and not f.startswith('__MACOSX')][:50]
    total_boxes = 0
    valid_boxes = 0
    for f in files:
        d = json.loads(z.read(f).decode('utf-8'))
        objs = d.get('objects', [])
        for o in objs:
            total_boxes += 1
            coord = o.get('obj_coord')
            if coord and len(coord) == 4:
                xmin, ymin, xmax, ymax = coord
                x1, y1, x2, y2 = xmin * 512, ymin * 512, xmax * 512, ymax * 512
                if x2 > x1 and y2 > y1:
                    valid_boxes += 1
            else:
                corner = o.get('obj_corner')
                if corner and len(corner) == 8:
                    xs = [corner[i] * 512 for i in range(0, 8, 2)]
                    ys = [corner[i] * 512 for i in range(1, 8, 2)]
                    x1, x2 = min(xs), max(xs)
                    y1, y2 = min(ys), max(ys)
                    if x2 > x1 and y2 > y1:
                        valid_boxes += 1

print(f"Total boxes in 50 files: {total_boxes}, Valid boxes: {valid_boxes}")

