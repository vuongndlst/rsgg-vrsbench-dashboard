"""Ablation study on 100 COCO Karpathy images comparing:
- C0: Objects only
- C1-Zero: Baseline SGG (native) + Zero-shot Qwen
- C1-Few:  Baseline SGG (native) + Few-shot (K=2) Qwen
- C2-Zero: RSG-Former (feedback) + Zero-shot Qwen
- C2-Few:  RSG-Former (feedback) + Few-shot (K=2) Qwen
"""
import os, sys, json, io, time, gc
from pathlib import Path
import torch
from PIL import Image

# Ensure GPU 1 is targeted and private java 8 is on PATH
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
java_bin = os.path.expanduser('~/rsg-former/.tools/java/temurin8-52dcd578baca/jdk8u504-b01-jre/bin')
if os.path.exists(java_bin):
    os.environ['PATH'] = java_bin + ':' + os.environ['PATH']

ROOT = Path.home() / 'rsg-former'
SHARED = Path.home() / 'shared-assets/datasets'
sys.path.insert(0, str(ROOT / '.tools/deps'))
sys.path.insert(0, str(ROOT / 'src'))

from rsgformer.vg_v2_runtime import native_model, capture, Adaptation
from rsgformer.vg_native_postprocess import native_vg_postprocess
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider

START_TIME = time.time()
print(f"[{time.strftime('%X')}] Starting 5-Arm Captioning Ablation on 100 COCO Karpathy images...")

# 1. Load manifest and select 100 images
manifest_path = SHARED / 'coco_karpathy/test_manifest.json'
manifest = json.loads(manifest_path.read_text())
test_images = manifest['images'][:100]
print(f"Selected {len(test_images)} test images.")

# Load VG150 vocabularies
meta = json.loads((ROOT / 'data/vg150/VG-SGG-dicts-with-attri.json').read_text())
obj_names = meta['idx_to_label']
pred_names = meta['idx_to_predicate']

def extract_triplets_and_objects(pred, eq):
    triplets = []
    seen_triples = set()
    detected_objects = set()
    
    for q_idx in range(len(pred['pred_boxes'])):
        cls_id = int(pred['pred_classes'][q_idx]) + 1
        name = obj_names.get(str(cls_id), f'obj_{cls_id}')
        detected_objects.add(name)
        
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

# 2. SGG Inference: Extract C1 (native baseline) and C2 (feedback RSG-Former)
c1_graphs = {}
c2_graphs = {}
objects_cache = {}

print(f"[{time.strftime('%X')}] Loading base detector model...")
base_model, _, _ = native_model(ROOT)

# Extract C1 (Native Baseline)
ckpt_c1 = ROOT / 'artifacts/vg_method_v2_speed_m2_v1/confirmation_native_1729/best.pt'
ck1 = torch.load(ckpt_c1, map_location='cpu', weights_only=True)
base_model.load_state_dict(ck1['native_head'], strict=False)
adapt_c1 = Adaptation(base_model, 'native').cuda().eval()
base_model.eval()

print(f"[{time.strftime('%X')}] Extracting C1 (Native Baseline) Scene Graphs...")
for idx, entry in enumerate(test_images):
    img_path = SHARED / 'coco_karpathy/images' / entry['file_name']
    image = Image.open(img_path).convert('RGB')
    with torch.no_grad():
        item, full = capture(base_model, image)
        batch = {k: item[k][None].float().cuda() for k in ['nodes', 'boxes', 'logits', 'visual']}
        out, _ = adapt_c1(batch)
        pred = native_vg_postprocess(out, *image.size)
    triplets, objs = extract_triplets_and_objects(pred, pred['query_to_nms_equivalence_id'])
    c1_graphs[entry['id']] = triplets
    objects_cache[entry['id']] = objs

del adapt_c1, ck1
gc.collect()
torch.cuda.empty_cache()

# Extract C2 (RSG-Former Feedback)
ckpt_c2 = ROOT / 'artifacts/vg_method_v2_speed_m2_v1/confirmation_feedback_1729/best.pt'
ck2 = torch.load(ckpt_c2, map_location='cpu', weights_only=True)
base_model.load_state_dict(ck2['native_head'], strict=False)
adapt_c2 = Adaptation(base_model, 'feedback')
adapt_c2.residual.load_state_dict(ck2['residual'], strict=True)
for p in adapt_c2.parameters(): p.requires_grad_(False)
adapt_c2 = adapt_c2.cuda().eval()

print(f"[{time.strftime('%X')}] Extracting C2 (RSG-Former Feedback) Scene Graphs...")
for idx, entry in enumerate(test_images):
    img_path = SHARED / 'coco_karpathy/images' / entry['file_name']
    image = Image.open(img_path).convert('RGB')
    with torch.no_grad():
        item, full = capture(base_model, image)
        batch = {k: item[k][None].float().cuda() for k in ['nodes', 'boxes', 'logits', 'visual']}
        out, _ = adapt_c2(batch)
        pred = native_vg_postprocess(out, *image.size)
    triplets, _ = extract_triplets_and_objects(pred, pred['query_to_nms_equivalence_id'])
    c2_graphs[entry['id']] = triplets

del base_model, adapt_c2, ck2, item, full, batch, out, pred
gc.collect()
torch.cuda.empty_cache()
print(f"[{time.strftime('%X')}] SGG extractions complete. VRAM freed.")

# 3. Load Qwen-14B
from transformers import AutoTokenizer, AutoModelForCausalLM
model_path = '/home/stackops/etechs/msvd-tkg-captioning/models/qwen2.5-14b'
print(f"[{time.strftime('%X')}] Loading Qwen-14B...")
tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
qwen = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map='auto',
    local_files_only=True
)
qwen.eval()
print(f"[{time.strftime('%X')}] Qwen-14B loaded.")

# Prompt definitions
sys_zero = (
    "You are an expert image captioner. You are given visual evidence detected in an image. "
    "Write exactly ONE concise, natural English sentence (8 to 15 words) describing the primary activity or scene. "
    "Do not include object IDs, lists, bullet points, or metadata. Be factual and fluent."
)

few_shot_demos = (
    "EXAMPLE 1:\n"
    "Detected visual relations: dog playing with ball; dog running on grass; ball in grass\n"
    "Description: A brown dog running across the green grass with a ball in its mouth.\n\n"
    "EXAMPLE 2:\n"
    "Detected visual relations: person holding umbrella; person walking on sidewalk; car on street\n"
    "Description: A person walking down a city sidewalk under an umbrella in the rain.\n\n"
)

def format_triplets(triplets, fallback_objs):
    if triplets:
        return '; '.join([f"{t['subject']} {t['predicate']} {t['object']}" for t in triplets])
    return ', '.join(fallback_objs[:5])

def generate_caption(user_content, sys_content=sys_zero):
    msg = [{'role': 'system', 'content': sys_content}, {'role': 'user', 'content': user_content}]
    text = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    inputs = tok([text], return_tensors='pt').to('cuda')
    with torch.no_grad():
        out = qwen.generate(**inputs, max_new_tokens=32, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

arms = {
    'C0_objects': {},
    'C1_native_zero': {},
    'C1_native_few': {},
    'C2_rsg_zero': {},
    'C2_rsg_few': {}
}

print(f"[{time.strftime('%X')}] Generating captions across all 5 arms for 100 images...")
for idx, entry in enumerate(test_images):
    img_id = entry['id']
    objs_str = ', '.join(objects_cache[img_id][:8])
    c1_str = format_triplets(c1_graphs[img_id], objects_cache[img_id])
    c2_str = format_triplets(c2_graphs[img_id], objects_cache[img_id])
    
    # C0
    arms['C0_objects'][img_id] = generate_caption(
        f"Detected objects: {objs_str}\nWrite one sentence describing the whole image:"
    )
    # C1 Zero
    arms['C1_native_zero'][img_id] = generate_caption(
        f"Detected visual relations: {c1_str}\nWrite one sentence describing the whole image:"
    )
    # C1 Few-shot
    arms['C1_native_few'][img_id] = generate_caption(
        few_shot_demos + f"Detected visual relations: {c1_str}\nDescription:"
    )
    # C2 Zero
    arms['C2_rsg_zero'][img_id] = generate_caption(
        f"Detected visual relations: {c2_str}\nWrite one sentence describing the whole image:"
    )
    # C2 Few-shot
    arms['C2_rsg_few'][img_id] = generate_caption(
        few_shot_demos + f"Detected visual relations: {c2_str}\nDescription:"
    )
    
    if (idx + 1) % 25 == 0:
        print(f"  Completed {idx + 1}/100 images...")

del qwen, tok
gc.collect()
torch.cuda.empty_cache()
print(f"[{time.strftime('%X')}] Caption generation completed.")

# 4. Evaluate with pycocoevalcap
print(f"[{time.strftime('%X')}] Evaluating all 5 arms with pycocoevalcap...")
ref_path = SHARED / 'coco_karpathy/test_references_coco.json'
refs_data = json.loads(ref_path.read_text())

target_ids = set(c2_graphs.keys())
gts_raw = {}
for ann in refs_data['annotations']:
    if ann['image_id'] in target_ids:
        gts_raw.setdefault(ann['image_id'], []).append({'caption': ann['caption']})

tokenizer = PTBTokenizer()
gts_tok = tokenizer.tokenize(gts_raw)

def evaluate_preds(preds_dict):
    preds_raw = {k: [{'caption': v}] for k, v in preds_dict.items()}
    res_tok = tokenizer.tokenize(preds_raw)
    scores = {}
    
    b_score, _ = Bleu(4).compute_score(gts_tok, res_tok)
    for i, s in enumerate(b_score, 1): scores[f'Bleu_{i}'] = float(s)
    
    meteor_scorer = Meteor()
    try:
        m_score, _ = meteor_scorer.compute_score(gts_tok, res_tok)
        scores['METEOR'] = float(m_score)
    finally:
        if hasattr(meteor_scorer, 'lock') and meteor_scorer.lock.locked():
            meteor_scorer.lock.release()
        del meteor_scorer
        
    r_score, _ = Rouge().compute_score(gts_tok, res_tok)
    scores['ROUGE_L'] = float(r_score)
    
    c_score, _ = Cider().compute_score(gts_tok, res_tok)
    scores['CIDEr'] = float(c_score)
    return scores

results = {}
for arm_name, preds in arms.items():
    s = evaluate_preds(preds)
    results[arm_name] = {k: v * 100 for k, v in s.items()}
    print(f"  {arm_name:15s} -> CIDEr: {results[arm_name]['CIDEr']:.2f}, BLEU-4: {results[arm_name]['Bleu_4']:.2f}, METEOR: {results[arm_name]['METEOR']:.2f}")

report = {
    'status': 'COCO_PILOT100_5ARM_ABLATION_COMPLETE',
    'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    'wall_seconds': time.time() - START_TIME,
    'metrics': results,
    'samples': [
        {
            'image_id': entry['id'],
            'file_name': entry['file_name'],
            'C1_triplets': c1_graphs[entry['id']],
            'C2_triplets': c2_graphs[entry['id']],
            'caption_C0': arms['C0_objects'][entry['id']],
            'caption_C1_zero': arms['C1_native_zero'][entry['id']],
            'caption_C1_few': arms['C1_native_few'][entry['id']],
            'caption_C2_zero': arms['C2_rsg_zero'][entry['id']],
            'caption_C2_few': arms['C2_rsg_few'][entry['id']],
            'references': [a['caption'].strip() for a in gts_raw[entry['id']]]
        }
        for entry in test_images[:10]
    ]
}

out_path = ROOT / 'reports/COCO_PILOT100_5ARM_ABLATION_RESULT.json'
out_path.write_text(json.dumps(report, indent=2))
print(f"[{time.strftime('%X')}] Report saved to {out_path}")
print("SUMMARY_JSON_START")
print(json.dumps({'metrics': results, 'wall_seconds': round(report['wall_seconds'], 1)}, indent=2))
print("SUMMARY_JSON_END")

