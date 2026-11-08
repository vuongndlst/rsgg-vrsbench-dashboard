"""Progressive RSG-Former Scene Graph Extractor for COCO 2014 Train Split (82,783 images).
Features:
- Fixed to GPU 1 (CUDA_VISIBLE_DEVICES=1)
- Resumable: skips images already processed in existing chunks
- Saves in safe chunks (5,000 images per chunk) to artifacts/sgg_train_cache/
- Formats each entry with 'id', 'file_name', 'triplets', 'objects', 'rel_str', 'captions'
"""
import os, sys, json, time, gc
from pathlib import Path
import torch
from PIL import Image

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

ROOT = Path.home() / 'rsg-former'
SHARED = Path.home() / 'shared-assets/datasets/coco_karpathy'
sys.path.insert(0, str(ROOT / '.tools/deps'))
sys.path.insert(0, str(ROOT / 'src'))

from rsgformer.vg_v2_runtime import native_model, capture, Adaptation
from rsgformer.vg_native_postprocess import native_vg_postprocess

CACHE_DIR = ROOT / 'artifacts/sgg_train_cache'
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CHUNK_SIZE = 5000

print(f"[{time.strftime('%X')}] Starting Progressive RSG-Former SGG Extractor...")

# 1. Load dataset_coco.json
coco_json_path = SHARED / 'annotations/dataset_coco.json'
assert coco_json_path.exists(), f"Cannot find {coco_json_path}"
print(f"Loading {coco_json_path}...")
with open(coco_json_path, 'r') as f:
    coco_data = json.load(f)

train_images = [img for img in coco_data['images'] if img.get('split') == 'train']
print(f"Total train split images: {len(train_images)}")

# Check existing chunks to resume
processed_ids = set()
for chunk_file in CACHE_DIR.glob('chunk_*.json'):
    try:
        cdata = json.loads(chunk_file.read_text())
        for item in cdata:
            processed_ids.add(item['id'])
    except Exception as e:
        print(f"Warning reading {chunk_file}: {e}")

print(f"Already processed in previous runs: {len(processed_ids)} images")
remaining_images = [img for img in train_images if img['cocoid'] not in processed_ids]
print(f"Remaining images to process: {len(remaining_images)}")

if not remaining_images:
    print("All train images already processed! Exiting.")
    sys.exit(0)

# 2. Load Label Dictionaries
meta = json.loads((ROOT / 'data/vg150/VG-SGG-dicts-with-attri.json').read_text())
obj_names = meta['idx_to_label']
pred_names = meta['idx_to_predicate']

def extract_triplets_and_objects(pred, eq):
    triplets = []
    seen_triples = set()
    detected_objects = set()
    
    for q_idx in range(len(pred['pred_boxes'])):
        cls_id = int(pred['pred_classes'][q_idx]) + 1
        detected_objects.add(obj_names.get(str(cls_id), f'obj_{cls_id}'))
        
    for rank, ((s, o), scores, score) in enumerate(zip(pred['pred_rel_inds'], pred['rel_scores'], pred['ranking_scores']), 1):
        sid, oid = int(eq[s]), int(eq[o])
        label = int(scores.argmax()) + 1
        if sid == oid or score <= 0: continue
        s_cls = int(pred['pred_classes'][s]) + 1
        o_cls = int(pred['pred_classes'][o]) + 1
        s_name = obj_names.get(str(s_cls), f'obj_{s_cls}')
        o_name = obj_names.get(str(o_cls), f'obj_{o_cls}')
        p_name = pred_names.get(str(label), f'pred_{label}')
        
        triple_key = (s_name, p_name, o_name)
        if triple_key in seen_triples: continue
        seen_triples.add(triple_key)
        triplets.append({'subject': s_name, 'predicate': p_name, 'object': o_name, 'score': float(score)})
        if len(triplets) >= 8: break
        
    return triplets, sorted(detected_objects)[:10]

# 3. Load RSG-Former Detector Model
print("Loading RSG-Former native detector model...")
base_model, _, _ = native_model(ROOT)
ckpt_path = ROOT / 'artifacts/vg_method_v2_speed_m2_v1/confirmation_native_1729/best.pt'
ck = torch.load(ckpt_path, map_location='cpu', weights_only=True)
base_model.load_state_dict(ck['native_head'], strict=False)
adapt = Adaptation(base_model, 'native').cuda().eval()
base_model.eval()

img_base_dir = SHARED / 'images/train2014'

# 4. Process in Chunks
current_chunk = []
existing_chunks = list(CACHE_DIR.glob('chunk_*.json'))
chunk_idx = len(existing_chunks) + 1
t0 = time.time()

print(f"Beginning inference on {len(remaining_images)} images...")
for idx, entry in enumerate(remaining_images, 1):
    img_path = img_base_dir / entry['filename']
    if not img_path.exists():
        # Fallback to images/
        img_path = SHARED / 'images' / entry['filename']
    if not img_path.exists():
        continue
    
    try:
        image = Image.open(img_path).convert('RGB')
        with torch.no_grad():
            item, full = capture(base_model, image)
            batch = {k: item[k][None].float().cuda() for k in ['nodes', 'boxes', 'logits', 'visual']}
            out, _ = adapt(batch)
            pred = native_vg_postprocess(out, *image.size)
        triplets, objs = extract_triplets_and_objects(pred, pred['query_to_nms_equivalence_id'])
        
        if triplets:
            rel_str = '; '.join([f"{t['subject']} {t['predicate']} {t['object']}" for t in triplets[:8]])
        else:
            rel_str = ', '.join(objs[:5])
        
        captions = [s['raw'].strip() for s in entry.get('sentences', [])]
        
        current_chunk.append({
            'id': entry['cocoid'],
            'file_name': entry['filename'],
            'triplets': triplets,
            'objects': objs,
            'rel_str': rel_str,
            'captions': captions
        })
    except Exception as e:
        print(f"Error on image {entry['cocoid']}: {e}")
        continue
    
    # Save chunk when full
    if len(current_chunk) >= CHUNK_SIZE or idx == len(remaining_images):
        save_path = CACHE_DIR / f"chunk_{chunk_idx:03d}.json"
        save_path.write_text(json.dumps(current_chunk, indent=2))
        elapsed = time.time() - t0
        rate = idx / max(elapsed, 0.01)
        print(f"[{time.strftime('%X')}] Saved chunk {chunk_idx} ({len(current_chunk)} images) -> {save_path.name} | Total: {idx}/{len(remaining_images)} ({idx/len(remaining_images)*100:.1f}%) | Speed: {rate:.2f} img/s")
        chunk_idx += 1
        current_chunk = []

print("Progressive SGG Extraction COMPLETE!")

