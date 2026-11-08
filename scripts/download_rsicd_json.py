import urllib.request
import os

url = "https://raw.githubusercontent.com/201528014227051/RSICD_optimal/master/dataset_rsicd.json"
out_dir = os.path.join(os.path.expanduser("~"), "rsg-former/data/rsicd")
os.makedirs(out_dir, exist_ok=True)
out_file = os.path.join(out_dir, "dataset_rsicd.json")

print(f"Downloading {url} to {out_file}...")
try:
    urllib.request.urlretrieve(url, out_file)
    print(f"Downloaded successfully! File size: {os.path.getsize(out_file)/(1024*1024):.2f} MB")
except Exception as e:
    print("Download error:", e)

