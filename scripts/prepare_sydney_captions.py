import os, zipfile, json
from pathlib import Path

ROOT = Path.home() / 'rsg-former'
target_dir = ROOT / 'data' / 'sydney_captions'
target_dir.mkdir(parents=True, exist_ok=True)

# Find Sydney_captions.zip in HF cache
cache_dir = Path.home() / '.cache' / 'huggingface' / 'hub'
zip_files = list(cache_dir.rglob('Sydney_captions.zip'))
print("Found zip files:", zip_files)

if zip_files:
    src_zip = zip_files[0]
    print(f"Extracting {src_zip} to {target_dir}...")
    with zipfile.ZipFile(src_zip, 'r') as z:
        z.extractall(target_dir)
    print("Extraction complete!")
    
    # Check contents
    files = list(target_dir.rglob('*'))
    print(f"Total extracted items: {len(files)}")
    json_path = target_dir / 'dataset.json'
    if json_path.exists():
        data = json.loads(json_path.read_text(encoding='utf-8'))
        print("dataset.json keys:", list(data.keys()) if isinstance(data, dict) else f"list of {len(data)}")
        if isinstance(data, dict) and 'images' in data:
            print("Number of images in dataset.json:", len(data['images']))
            print("First image item sample:", data['images'][0])
        elif isinstance(data, list):
            print("First item sample:", data[0])
    img_dir = target_dir / 'images'
    if img_dir.exists():
        imgs = list(img_dir.glob('*.tif')) + list(img_dir.glob('*.png')) + list(img_dir.glob('*.jpg'))
        print(f"Total images found in {img_dir}: {len(imgs)}")
else:
    print("Sydney_captions.zip not found!")

