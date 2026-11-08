import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from pathlib import Path
import json

ROOT = Path.home() / 'rsg-former'
ucm_dir = ROOT / 'data' / 'ucm_captions'
ucm_dir.mkdir(parents=True, exist_ok=True)
img_dir = ucm_dir / 'images'
img_dir.mkdir(parents=True, exist_ok=True)

repo = "caglarmert/UCM-captions"
splits = [("train", "data/train-00000-of-00001.parquet"),
          ("val", "data/validation-00000-of-00001.parquet"),
          ("test", "data/test-00000-of-00001.parquet")]

all_images_meta = []
img_counter = 0
sent_counter = 0

print("Extracting UCM-Captions images and building dataset.json...")
for split_name, rel_path in splits:
    parquet_path = hf_hub_download(repo_id=repo, filename=rel_path, repo_type="dataset")
    table = pq.read_table(parquet_path)
    num_rows = table.num_rows
    print(f"Processing split {split_name}: {num_rows} images...")
    
    for i in range(num_rows):
        fname = table['file_name'][i].as_py()
        img_bytes = table['image'][i].as_py()['bytes']
        
        # Save image
        out_img_path = img_dir / fname
        if not out_img_path.exists():
            out_img_path.write_bytes(img_bytes)
            
        # Collect sentences
        sentences = []
        sentids = []
        for c_idx in range(1, 6):
            cap_key = f"caption_{c_idx}"
            cap_text = table[cap_key][i].as_py().strip()
            sentences.append({
                "tokens": cap_text.split(),
                "raw": cap_text,
                "imgid": img_counter,
                "sentid": sent_counter
            })
            sentids.append(sent_counter)
            sent_counter += 1
            
        all_images_meta.append({
            "filename": fname,
            "imgid": img_counter,
            "split": split_name,
            "sentids": sentids,
            "sentences": sentences
        })
        img_counter += 1

dataset_dict = {
    "images": all_images_meta,
    "dataset": "ucm_captions",
    "total_images": len(all_images_meta),
    "total_sentences": sent_counter
}

json_path = ucm_dir / 'dataset.json'
json_path.write_text(json.dumps(dataset_dict, indent=2), encoding='utf-8')
print(f"SUCCESS: Saved {len(all_images_meta)} images and dataset.json ({json_path.stat().st_size / (1024*1024):.2f} MB)")
print("Sample extracted image:", list(img_dir.glob('*.jpg'))[:3])

