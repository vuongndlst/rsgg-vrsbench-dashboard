"""Extract SGG triplets for the 503 validation images to build the training dataset.
Pipeline:
  1. Load SGG native detector model on GPU 1
  2. Extract scene graphs for all 503 images (~3 mins)
  3. Format into (Triplets, Image, Captions) training pairs
  4. Save to artifacts/coco_val500_sgg_train_dataset.json
"""
import os, sys, json, time, gc
from pathlib import Path
import torch
from PIL import Image

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

ROOT = Path.home() / 'rsg-former'
SHARED = Path.home() / 'shared-assets/datasets'
sys.path.insert(0, str(ROOT / '.tools/deps'))
sys.path.insert(0, str(ROOT / 'src'))

from rsgformer.vg_v2_runtime import native_model, capture, Adaptation
from rsgformer.vg_native_postprocess import native_vg_postprocess

manifest_path = ROOT / 'data/coco_val500_train_manifest.json'
manifest = json.loads(manifest_path.read_text())
print(f"Loaded {len(manifest)} images from {manifest_path}")

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

print("Loading base detector model...")
base_model, _, _ = native_model(ROOT)
ckpt_path = ROOT / 'artifacts/vg_method_v2_speed_m2_v1/confirmation_native_1729/best.pt'
ck = torch.load(ckpt_path, map_location='cpu', weights_only=True)
base_model.load_state_dict(ck['native_head'], strict=False)
adapt = Adaptation(base_model, 'native').cuda().eval()
base_model.eval()

dataset = []
t0 = time.time()
print(f"Extracting SGG for {len(manifest)} images...")

for idx, entry in enumerate(manifest, 1):
    img_path = SHARED / 'coco_karpathy/images' / entry['file_name']
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
        
    dataset.append({
        'id': entry['id'],
        'file_name': entry['file_name'],
        'triplets': triplets,
        'objects': objs,
        'rel_str': rel_str,
        'captions': entry['captions']
    })
    
    if idx % 100 == 0 or idx == len(manifest):
        elapsed = time.time() - t0
        rate = idx / elapsed
        print(f"  Processed {idx}/{len(manifest)} ({idx/len(manifest)*100:.1f}%) | Speed: {rate:.2f} img/s")

del base_model, adapt, ck, item, full, batch, out, pred
gc.collect()
torch.cuda.empty_cache()

out_file = ROOT / 'artifacts/coco_val500_sgg_train_dataset.json'
out_file.parent.mkdir(parents=True, exist_ok=True)
out_file.write_text(json.dumps(dataset, indent=2))
print(f"Extraction complete in {time.time() - t0:.1f}s. Saved {len(dataset)} records to {out_file}")

