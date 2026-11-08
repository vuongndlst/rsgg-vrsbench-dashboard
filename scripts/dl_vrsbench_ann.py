import urllib.request, zipfile
from pathlib import Path

target_dir = Path.home() / 'shared-assets/datasets/vrsbench'
ann_zip = target_dir / 'Annotations_train.zip'

if not ann_zip.exists():
    url = "https://huggingface.co/datasets/xiang709/VRSBench/resolve/main/Annotations_train.zip"
    print(f"Downloading {url}...")
    urllib.request.urlretrieve(url, ann_zip)
    print("Downloaded!")

with zipfile.ZipFile(ann_zip) as z:
    names = z.namelist()
    print(f"Total files in Annotations_train.zip: {len(names)}")
    print("Sample filenames:", names[:10])
    # Read sample content
    sample_text = z.read(names[0]).decode('utf-8')
    print(f"\nContent of {names[0]}:\n", sample_text[:400])

