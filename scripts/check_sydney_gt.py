import json, os

p = os.path.join(os.path.expanduser("~"), "rsg-former/data/sydney_captions/dataset.json")
if os.path.exists(p):
    data = json.load(open(p))
    images = data.get("images", [])
    print(f"Total images: {len(images)}")
    for img in images[:5]:
        print(f"Img ID: {img.get('imgid')}, Filename: {img.get('filename')}, Split: {img.get('split')}")
        for s in img.get("sentences", []):
            print("  GT:", s.get("raw"))

