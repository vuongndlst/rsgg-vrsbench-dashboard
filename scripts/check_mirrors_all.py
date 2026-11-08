import subprocess

candidates = [
    "http://images.cocodataset.org/zips/train2014.zip",
    "https://pjreddie.com/media/files/train2014.zip",
    "https://storage.googleapis.com/coco-dataset/train2014.zip",
    "http://msvocods.blob.core.windows.net/coco2014/train2014.zip",
    "https://data.deepai.org/coco2014train.zip"
]

for c in candidates:
    out = subprocess.getoutput(f"curl -s -I --connect-timeout 2 --max-time 3 '{c}' | head -n 3")
    print(c, "->", out.replace('\n', ' | '))

