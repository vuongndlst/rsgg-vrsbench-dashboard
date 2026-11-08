import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from pathlib import Path
import json

ROOT = Path.home() / 'rsg-former'
ucm_dir = ROOT / 'data' / 'ucm_captions'
ucm_dir.mkdir(parents=True, exist_ok=True)
img_dir = ucm_dir / 'images'
img_dir.mkdir(parents=True, exist_ok=True)

# 1. Download train, validation, test
repo = "caglarmert/UCM-captions"
for split in ["test", "val", "train"]:
    fname = f"data/{'validation' if split=='val' else split}-00000-of-00001.parquet"
    print(f"Downloading UCM {split}...")
    local_p = hf_hub_download(repo_id=repo, filename=fname, repo_type="dataset")
    print(f"  Downloaded: {local_p}")
    
    # Read table
    table = pq.read_table(local_p)
    print(f"  {split} rows: {table.num_rows}, columns: {table.schema.names}")

# Read test table sample
test_p = hf_hub_download(repo_id=repo, filename="data/test-00000-of-00001.parquet", repo_type="dataset")
table = pq.read_table(test_p)
print("Test schema details:")
for f in table.schema:
    print(f"  {f.name}: {f.type}")
row0 = {col: table[col][0].as_py() for col in table.schema.names}
print("Row 0 non-binary fields:")
for k, v in row0.items():
    if k != 'image':
        print(f"  {k}: {v}")
    else:
        print(f"  image type: {type(v)}, bytes: {len(v['bytes']) if isinstance(v, dict) and 'bytes' in v else 'no bytes'}")

