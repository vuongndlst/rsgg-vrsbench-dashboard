from huggingface_hub import hf_hub_download
from pathlib import Path
import zipfile, os

ROOT = Path.home() / 'rsg-former'
dior_dir = ROOT / 'data' / 'dior_rsvg'
dior_dir.mkdir(parents=True, exist_ok=True)

print("Downloading DIOR-RSVG Annotations.zip and test.txt...")
ann_path = hf_hub_download(repo_id="LittleCollections/DIOR-RSVG", filename="Annotations.zip", repo_type="dataset")
test_path = hf_hub_download(repo_id="LittleCollections/DIOR-RSVG", filename="test.txt", repo_type="dataset")

print(f"Downloaded Annotations: {ann_path}")
print(f"Downloaded test.txt: {test_path}")

# Extract Annotations
with zipfile.ZipFile(ann_path, 'r') as z:
    z.extractall(dior_dir)
print("Extracted Annotations to:", dior_dir)

# Copy test.txt
out_test = dior_dir / 'test.txt'
out_test.write_bytes(Path(test_path).read_bytes())
print("Saved test.txt to:", out_test)

# Check files
ann_files = list(dior_dir.rglob('*'))
print(f"Total DIOR items in {dior_dir}: {len(ann_files)}")
sample_ann = [f for f in ann_files if f.is_file() and f.suffix in ['.json', '.xml', '.txt']][:5]
for s in sample_ann:
    print(f"  sample: {s.name} ({s.stat().st_size} bytes)")

