import json, os

p = os.path.join(os.path.expanduser("~"), "rsg-former/data/rsicd/dataset_rsicd.json")
data = json.load(open(p))
images = data.get("images", [])
print(f"Total RSICD images: {len(images)}")

splits = {}
for img in images:
    sp = img.get("split", "unknown")
    splits[sp] = splits.get(sp, 0) + 1

print(f"Splits: {splits}")
print("\nFirst 3 images:")
for img in images[:3]:
    print(f"  imgid: {img.get('imgid')}, filename: {img.get('filename')}, split: {img.get('split')}")
    for s in img.get("sentences", [])[:2]:
        print(f"    raw: {s.get('raw')}")

