from huggingface_hub import hf_hub_download
from pathlib import Path

print("Downloading UCM test parquet...")
f = hf_hub_download(repo_id="caglarmert/UCM-captions", filename="data/test-00000-of-00001.parquet", repo_type="dataset")
print("Downloaded to:", f, "Size:", Path(f).stat().st_size / (1024*1024), "MB")

