"""Pilot 100-image captioning evaluation on COCO Karpathy test using RSG-Former + Qwen-14B."""
import os, sys, json, io, time, gc, subprocess, tempfile
from pathlib import Path
import torch
from PIL import Image

# Ensure GPU 1 is targeted and java 8 is on PATH
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
from pycocotools.coco import COCO
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider

START_TIME = time.time()
print(f"[{time.strftime('%X')}] Starting COCO Karpathy 100-image captioning pilot...")

# 1. Load manifest and select 100 images
manifest_path = SHARED / 'coco_karpathy/test_manifest.json'
manifest = json.loads(manifest_path.read_text())
test_images = manifest['images'][:100]
print(f"Selected {len(test_images)} test images from COCO Karpathy.")

# Load VG150 vocabularies
meta = json.loads((ROOT / 'data/vg150/VG-SGG-dicts-with-attri.json').read_text())
obj_names = meta['idx_to_label']
pred_names = meta['idx_to_predicate']

# 2. Extract Scene Graphs with RSG-Former
print(f"[{time.strftime('%X')}] Loading RSG-Former detector and adaptation model on GPU 1...")
sgg_model, criterion, config = native_model(ROOT)
ckpt_path = ROOT / 'artifacts/vg_method_v2_speed_m2_v1/confirmation_context_1729/best.pt'
ck = torch.load(ckpt_path, map_location='cpu', weights_only=True)
sgg_model.load_state_dict(ck['native_head'], strict=False)
adapt = Adaptation(sgg_model, 'context')
adapt.residual.load_state_dict(ck['residual'], strict=True)
for p in adapt.parameters(): p.requires_grad_(False)
adapt.cuda().eval()
sgg_model.eval()

extracted_graphs = []
print(f"[{time.strftime('%X')}] Running Scene Graph inference on 100 images...")
for idx, entry in enumerate(test_images):
    img_id = entry['id']
    img_path = SHARED / 'coco_karpathy/images' / entry['file_name']
    image = Image.open(img_path).convert('RGB')
    
    with torch.no_grad():
        item, full = capture(sgg_model, image)
        batch = {k: item[k][None].float().cuda() for k in ['nodes', 'boxes', 'logits', 'visual']}
        out, _ = adapt(batch)
        pred = native_vg_postprocess(out, *image.size)
        
    eq = pred['query_to_nms_equivalence_id']
    triplets = []
    seen_triples = set()
    detected_objects = set()
    
    # Collect detected objects
    for q_idx in range(len(pred['pred_boxes'])):
        cls_id = int(pred['pred_classes'][q_idx]) + 1
        name = obj_names.get(str(cls_id), f'obj_{cls_id}')
        detected_objects.add(name)
        
    # Collect top salient relation triplets
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
        
    extracted_graphs.append({
        'id': img_id,
        'file_name': entry['file_name'],
        'detected_objects': sorted(detected_objects)[:10],
        'triplets': triplets
    })
    if (idx + 1) % 25 == 0:
        print(f"  Processed {idx + 1}/100 images...")

# Free SGG model from VRAM before loading Qwen
del sgg_model, adapt, ck, item, full, batch, out, pred
gc.collect()
torch.cuda.empty_cache()
print(f"[{time.strftime('%X')}] SGG extraction completed. VRAM freed.")

# 3. Load Qwen-14B for Caption Generation
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = '/home/stackops/etechs/msvd-tkg-captioning/models/qwen2.5-14b'
print(f"[{time.strftime('%X')}] Loading Qwen-14B model weights into GPU 1...")
tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
qwen = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map='auto',
    local_files_only=True
)
qwen.eval()
print(f"[{time.strftime('%X')}] Qwen-14B successfully loaded.")

# System prompts and verbalization templates
sys_prompt = (
    "You are an expert image captioner. You are given visual evidence detected in an image. "
    "Write exactly ONE concise, natural English sentence (8 to 15 words) describing the primary activity or scene. "
    "Do not include object IDs, lists, bullet points, or metadata. Be factual and fluent."
)

c0_captions = {} # Objects-only
c2_captions = {} # Full RSG-Former graph

print(f"[{time.strftime('%X')}] Generating captions for 100 images (Conditions C0 and C2)...")
for idx, item in enumerate(extracted_graphs):
    img_id = item['id']
    
    # C2: Triplets prompt
    if item['triplets']:
        triplets_str = '; '.join([f"{t['subject']} {t['predicate']} {t['object']}" for t in item['triplets']])
    else:
        triplets_str = ', '.join(item['detected_objects'][:5])
        
    user_msg_c2 = f"Detected visual relations: {triplets_str}\nWrite one sentence describing the whole image:"
    msg_c2 = [{'role': 'system', 'content': sys_prompt}, {'role': 'user', 'content': user_msg_c2}]
    text_c2 = tok.apply_chat_template(msg_c2, tokenize=False, add_generation_prompt=True)
    in_c2 = tok([text_c2], return_tensors='pt').to('cuda')
    
    with torch.no_grad():
        out_c2 = qwen.generate(**in_c2, max_new_tokens=32, do_sample=False, pad_token_id=tok.eos_token_id)
    cap_c2 = tok.decode(out_c2[0][in_c2.input_ids.shape[1]:], skip_special_tokens=True).strip()
    c2_captions[img_id] = cap_c2
    
    # C0: Objects-only prompt
    objects_str = ', '.join(item['detected_objects'][:8])
    user_msg_c0 = f"Detected objects: {objects_str}\nWrite one sentence describing the whole image:"
    msg_c0 = [{'role': 'system', 'content': sys_prompt}, {'role': 'user', 'content': user_msg_c0}]
    text_c0 = tok.apply_chat_template(msg_c0, tokenize=False, add_generation_prompt=True)
    in_c0 = tok([text_c0], return_tensors='pt').to('cuda')
    
    with torch.no_grad():
        out_c0 = qwen.generate(**in_c0, max_new_tokens=32, do_sample=False, pad_token_id=tok.eos_token_id)
    cap_c0 = tok.decode(out_c0[0][in_c0.input_ids.shape[1]:], skip_special_tokens=True).strip()
    c0_captions[img_id] = cap_c0
    
    if (idx + 1) % 25 == 0:
        print(f"  Generated {idx + 1}/100 pairs...")

del qwen, tok
gc.collect()
torch.cuda.empty_cache()
print(f"[{time.strftime('%X')}] Caption generation completed.")

# 4. Evaluation with pycocoevalcap
print(f"[{time.strftime('%X')}] Evaluating metrics on 100 images with pycocoevalcap...")
ref_path = SHARED / 'coco_karpathy/test_references_coco.json'
refs_data = json.loads(ref_path.read_text())

# Build ground truth dictionary for the 100 images
target_ids = set(c2_captions.keys())
gts_raw = {}
for ann in refs_data['annotations']:
    if ann['image_id'] in target_ids:
        gts_raw.setdefault(ann['image_id'], []).append({'caption': ann['caption']})

def evaluate_dict(preds_dict, gts_dict):
    tokenizer = PTBTokenizer()
    preds_raw = {k: [{'caption': v}] for k, v in preds_dict.items()}
    gts_tok = tokenizer.tokenize(gts_dict)
    res_tok = tokenizer.tokenize(preds_raw)
    
    scores = {}
    # Bleu
    bleu_scorer = Bleu(4)
    b_score, _ = bleu_scorer.compute_score(gts_tok, res_tok)
    for i, s in enumerate(b_score, 1):
        scores[f'Bleu_{i}'] = float(s)
    
    # METEOR
    meteor_scorer = Meteor()
    try:
        m_score, _ = meteor_scorer.compute_score(gts_tok, res_tok)
        scores['METEOR'] = float(m_score)
    finally:
        if hasattr(meteor_scorer, 'lock') and meteor_scorer.lock.locked():
            meteor_scorer.lock.release()
        del meteor_scorer
    
    # ROUGE
    rouge_scorer = Rouge()
    r_score, _ = rouge_scorer.compute_score(gts_tok, res_tok)
    scores['ROUGE_L'] = float(r_score)
    
    # CIDEr
    cider_scorer = Cider()
    c_score, _ = cider_scorer.compute_score(gts_tok, res_tok)
    scores['CIDEr'] = float(c_score)
    
    return scores

scores_c0 = evaluate_dict(c0_captions, gts_raw)
scores_c2 = evaluate_dict(c2_captions, gts_raw)

# Compile full report
report = {
    'status': 'COCO_KARPATHY_PILOT100_COMPLETE',
    'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    'num_images': len(test_images),
    'wall_seconds': time.time() - START_TIME,
    'metrics': {
        'C0_objects_only': {k: v * 100 for k, v in scores_c0.items()},
        'C2_rsg_former_triplets': {k: v * 100 for k, v in scores_c2.items()},
        'delta_C2_minus_C0': {k: (scores_c2[k] - scores_c0[k]) * 100 for k in scores_c2}
    },
    'samples': [
        {
            'image_id': item['id'],
            'file_name': item['file_name'],
            'triplets': item['triplets'],
            'caption_C0': c0_captions[item['id']],
            'caption_C2': c2_captions[item['id']],
            'references': [a['caption'].strip() for a in gts_raw[item['id']]]
        }
        for item in extracted_graphs[:10]
    ]
}

out_report_path = ROOT / 'reports/COCO_PILOT100_CAPTION_RESULT.json'
out_report_path.write_text(json.dumps(report, indent=2))
print(f"[{time.strftime('%X')}] Report saved to {out_report_path}")
print("SUMMARY_JSON_START")
print(json.dumps({
    'status': report['status'],
    'wall_seconds': round(report['wall_seconds'], 1),
    'C0_metrics': report['metrics']['C0_objects_only'],
    'C2_metrics': report['metrics']['C2_rsg_former_triplets'],
    'delta': report['metrics']['delta_C2_minus_C0'],
    'sample_comparison': report['samples'][0]
}, indent=2))
print("SUMMARY_JSON_END")
