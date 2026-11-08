import os
import urllib.request
import pandas as pd
import io
from PIL import Image

data_dir = os.path.join(os.path.expanduser("~"), "rsg-former/data/rsicd")
img_dir = os.path.join(data_dir, "images")
os.makedirs(img_dir, exist_ok=True)

parquet_path = os.path.join(data_dir, "test.parquet")
url = "https://huggingface.co/datasets/arampacha/rsicd/resolve/main/data/test-00000-of-00001.parquet"

if not os.path.exists(parquet_path):
    print(f"Downloading {url} ({52.53} MB)...")
    urllib.request.urlretrieve(url, parquet_path)
    print("Downloaded parquet successfully!")

df = pd.read_parquet(parquet_path)
print(f"Loaded parquet dataframe with {len(df)} rows. Columns: {df.columns.tolist()}")

# Extract images and metadata
extracted_count = 0
for idx, row in df.iterrows():
    raw_fn = row.get('filename', f"rsicd_{idx:05d}.jpg")
    fn = os.path.basename(raw_fn)
    img_out = os.path.join(img_dir, fn)
    if not os.path.exists(img_out):
        img_data = row.get('image')
        if isinstance(img_data, dict) and 'bytes' in img_data:
            with open(img_out, 'wb') as f:
                f.write(img_data['bytes'])
            extracted_count += 1
        elif hasattr(img_data, 'save'):
            img_data.save(img_out)
            extracted_count += 1

print(f"Extracted {extracted_count} images to {img_dir}. Total images in dir: {len(os.listdir(img_dir))}")
