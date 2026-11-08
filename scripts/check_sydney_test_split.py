import json, os

p = os.path.join(os.path.expanduser("~"), "rsg-former/data/sydney_captions/dataset.json")
data = json.load(open(p))
test_imgs = [img for img in data['images'] if img.get('split') == 'test']
print(f"Total test images: {len(test_imgs)}")

# Check filenames or classes
test_classes = set()
for img in test_imgs:
    fn = img.get('filename')
    # check sentences
    s0 = img.get('sentences', [])[0]['raw'] if img.get('sentences') else ''
    print(f"Test Img {img.get('imgid')}: {fn} -> GT 1st: {s0[:60]}")

