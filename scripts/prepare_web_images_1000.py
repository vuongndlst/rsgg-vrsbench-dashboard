"""Extract and compress the 1,000 VRSBench test images from Images_val.zip into WebP format
for the interactive GitHub Pages dashboard. Uses CPU only.
"""
import io, json, zipfile, time
from pathlib import Path
from PIL import Image

ROOT = Path.home() / 'rsg-former'
SHARED = Path.home() / 'shared-assets/datasets/vrsbench'

def main():
    t0 = time.time()
    eval_path = SHARED / 'VRSBench_EVAL_Cap.json'
    val_img_zip = SHARED / 'Images_val.zip'
    out_zip_path = ROOT / 'artifacts/web_images_1000.zip'
    out_zip_path.parent.mkdir(parents=True, exist_ok=True)

    with open(eval_path) as f:
        eval_data = json.load(f)

    target_samples = eval_data[:1000]
    unique_imgs = []
    seen = set()
    for item in target_samples:
        img_id = item.get('image_id', '')
        if img_id and img_id not in seen:
            seen.add(img_id)
            unique_imgs.append(img_id)

    print(f"Extracting {len(unique_imgs)} unique satellite images for the 1,000 test samples...", flush=True)

    count = 0
    with zipfile.ZipFile(val_img_zip, 'r') as iz, zipfile.ZipFile(out_zip_path, 'w', compression=zipfile.ZIP_STORED) as oz:
        namelist = set(iz.namelist())
        for img_id in unique_imgs:
            cand = f"Images_val/{img_id}" if f"Images_val/{img_id}" in namelist else img_id
            if cand not in namelist:
                continue
            raw = iz.read(cand)
            img = Image.open(io.BytesIO(raw)).convert('RGB')
            img.thumbnail((384, 384), Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            webp_name = img_id.replace('.png', '.webp').replace('.jpg', '.webp')
            img.save(buf, format='WEBP', quality=78, method=4)
            oz.writestr(webp_name, buf.getvalue())
            count += 1
            if count % 250 == 0:
                print(f"  Compressed {count}/{len(unique_imgs)} images...", flush=True)

    size_mb = out_zip_path.stat().st_size / (1024 * 1024)
    print(f"Done in {time.time()-t0:.1f}s! Saved {count} WebP images to {out_zip_path} ({size_mb:.2f} MB)", flush=True)

if __name__ == '__main__':
    main()

