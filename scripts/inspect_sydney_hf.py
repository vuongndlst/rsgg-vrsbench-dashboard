import urllib.request, json
from huggingface_hub import hf_hub_download

# Test downloading Sydney-Captions metadata / zip
path = hf_hub_download(repo_id="isaaccorley/Sydney-Captions", filename="Sydney_captions.zip", repo_type="dataset")
print("Sydney zip path:", path)
import zipfile
with zipfile.ZipFile(path, 'r') as z:
    names = z.namelist()
    print(f"Total files in Sydney zip: {len(names)}")
    print("Sample names:", names[:10])

