import subprocess

urls = [
    "http://images.cocodataset.org/zips/train2014.zip",
    "https://huggingface.co/datasets/HuggingFaceM4/COCO/resolve/main/data/train2014.zip",
    "https://hf-mirror.com/datasets/HuggingFaceM4/COCO/resolve/main/data/train2014.zip",
    "https://storage.googleapis.com/cross-llm-data/coco/train2014.zip"
]

for u in urls:
    res = subprocess.run(f"curl -s -I --connect-timeout 3 --max-time 5 '{u}' | head -n 5", shell=True, capture_output=True, text=True)
    print(f"URL: {u}")
    print(res.stdout.strip())
    print("-" * 40)

