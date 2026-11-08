import json, os

rep_p = os.path.join(os.path.expanduser("~"), "rsg-former/reports/sydney_detector_boxes_cache.json")
dets = json.load(open(rep_p))

test_ids = [218, 230, 263, 309, 355, 442, 538, 607]
for tid in test_ids:
    d = dets.get(str(tid), [])
    cats = [x['category'] for x in d]
    print(f"Test Img {tid}: {len(d)} detections -> {cats}")

