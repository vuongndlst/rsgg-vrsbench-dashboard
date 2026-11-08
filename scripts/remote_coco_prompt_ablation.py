"""Systematic Prompt Optimization Ablation on 100 COCO Karpathy images.
Reads cached SGG triplets from artifacts/coco_5000_sgg_native_cache.json.

Evaluates 6 prompt engineering strategies:
  - P0_baseline_few: Current K=2 few-shot prompt (Anchor)
  - P1_filtered_triplets: Noise-suppressed triplets (pruning anatomy/trivial has-part)
  - P2_participle_syntax: COCO-aligned participle phrasing ("A man riding..." vs "A man is riding...")
  - P3_hierarchical_prompt: Partitioned input (Core Action vs Context vs Attributes)
  - P4_diverse_k4: K=4 diverse prototypical demonstrations (motion, dining, animal, urban)
  - P5_combined_optimal: Filtered triplets + Participle syntax + K=4 diverse demonstrations
"""
import os, sys, json, io, time, gc
from pathlib import Path
import torch

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# Add private Java 8 to PATH for pycocoevalcap
java_bin = os.path.expanduser('~/rsg-former/.tools/java/temurin8-52dcd578baca/jdk8u504-b01-jre/bin')
if os.path.exists(java_bin):
    os.environ['PATH'] = java_bin + ':' + os.environ['PATH']

ROOT = Path.home() / 'rsg-former'
SHARED = Path.home() / 'shared-assets/datasets'
sys.path.insert(0, str(ROOT / '.tools/deps'))
sys.path.insert(0, str(ROOT / 'src'))

from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider

START_TIME = time.time()
print(f"[{time.strftime('%X')}] Starting Prompt Optimization Ablation...")

# 1. Load manifest (100 pilot images)
manifest_path = SHARED / 'coco_karpathy/test_manifest.json'
manifest = json.loads(manifest_path.read_text())
test_images = manifest['images'][:100]
test_ids = [entry['id'] for entry in test_images]
print(f"Loaded {len(test_images)} pilot test images.")

# 2. Load cached SGG extractions
sgg_cache_path = ROOT / 'artifacts/coco_5000_sgg_native_cache.json'
assert sgg_cache_path.exists(), "SGG cache missing!"
sgg_full_cache = json.loads(sgg_cache_path.read_text())
print(f"Loaded cached SGG data ({len(sgg_full_cache)} total images in cache).")

# Triplet filtering helper
TRIVIAL_PARTS = {'head', 'hair', 'eye', 'arm', 'leg', 'hand', 'ear', 'face', 'nose', 'mouth', 'foot', 'finger', 'toe'}
ACTION_PREDICATES = {'riding', 'eating', 'holding', 'playing with', 'walking on', 'sitting on', 'flying in', 'standing on', 'jumping on', 'driving', 'carrying'}

def filter_triplets(triplets):
    """Filter out trivial anatomical and self-part relations to elevate scene-level interactions."""
    filtered = []
    for t in triplets:
        s, p, o = t['subject'].lower(), t['predicate'].lower(), t['object'].lower()
        # Drop trivial 'has body part'
        if p == 'has' and (o in TRIVIAL_PARTS or s in TRIVIAL_PARTS):
            continue
        if p == 'of' and s in TRIVIAL_PARTS:
            continue
        filtered.append(t)
    return filtered if filtered else triplets

def format_triplets_raw(triplets, fallback_objs):
    if triplets:
        return '; '.join([f"{t['subject']} {t['predicate']} {t['object']}" for t in triplets[:8]])
    return ', '.join(fallback_objs[:5])

def format_triplets_filtered(triplets, fallback_objs):
    cleaned = filter_triplets(triplets)
    if cleaned:
        return '; '.join([f"{t['subject']} {t['predicate']} {t['object']}" for t in cleaned[:8]])
    return ', '.join(fallback_objs[:5])

def format_hierarchical(triplets, objs):
    cleaned = filter_triplets(triplets)
    actions = [f"{t['subject']} {t['predicate']} {t['object']}" for t in cleaned if t['predicate'] in ACTION_PREDICATES]
    spatial = [f"{t['subject']} {t['predicate']} {t['object']}" for t in cleaned if t['predicate'] not in ACTION_PREDICATES and t['predicate'] not in {'wearing'}]
    wearing = [f"{t['subject']} wearing {t['object']}" for t in cleaned if t['predicate'] == 'wearing']
    
    parts = []
    if actions: parts.append("Actions: " + '; '.join(actions[:4]))
    if spatial: parts.append("Context: " + '; '.join(spatial[:4]))
    if wearing: parts.append("Appearance: " + '; '.join(wearing[:2]))
    if not parts:
        parts.append("Visual entities: " + ', '.join(objs[:6]))
    return ' | '.join(parts)

# 3. Load Qwen-14B
from transformers import AutoTokenizer, AutoModelForCausalLM
model_path = '/home/stackops/etechs/msvd-tkg-captioning/models/qwen2.5-14b'
print(f"[{time.strftime('%X')}] Loading Qwen-14B...")
tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
if tok.pad_token_id is None:
    tok.pad_token_id = tok.eos_token_id
tok.padding_side = 'left'

qwen = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map='auto',
    local_files_only=True
)
qwen.eval()
print(f"[{time.strftime('%X')}] Qwen-14B loaded.")

# Prompt Definitions
SYS_STANDARD = (
    "You are an expert image captioner. You are given visual evidence detected in an image. "
    "Write exactly ONE concise, natural English sentence (8 to 15 words) describing the primary activity or scene. "
    "Do not include object IDs, lists, bullet points, or metadata. Be factual and fluent."
)

DEMO_K2_BASELINE = (
    "EXAMPLE 1:\n"
    "Detected visual relations: dog playing with ball; dog running on grass; ball in grass\n"
    "Description: A brown dog running across the green grass with a ball in its mouth.\n\n"
    "EXAMPLE 2:\n"
    "Detected visual relations: person holding umbrella; person walking on sidewalk; car on street\n"
    "Description: A person walking down a city sidewalk under an umbrella in the rain.\n\n"
)

DEMO_PARTICIPLE = (
    "EXAMPLE 1:\n"
    "Detected visual relations: man riding skateboard; skateboard on ramp; person watching man\n"
    "Description: A man riding a skateboard down a wooden ramp at a skate park.\n\n"
    "EXAMPLE 2:\n"
    "Detected visual relations: plane on runway; luggage cart near plane; building behind plane\n"
    "Description: A large commercial airplane parked on the airport runway near terminal buildings.\n\n"
)

DEMO_K4_DIVERSE = (
    "EXAMPLE 1 (Outdoor Action):\n"
    "Detected visual relations: man riding motorcycle; motorcycle on dirt road; trees behind road\n"
    "Description: A man riding a motorcycle down a dirt road surrounded by trees.\n\n"
    "EXAMPLE 2 (Indoor Dining):\n"
    "Detected visual relations: woman sitting at table; bowl of soup on table; woman holding spoon\n"
    "Description: A woman sitting at a table eating a bowl of soup with a spoon.\n\n"
    "EXAMPLE 3 (Animals/Nature):\n"
    "Detected visual relations: zebra standing in grass; zebras near tree; mountain in background\n"
    "Description: A herd of zebras grazing in a dry grass field near mountains.\n\n"
    "EXAMPLE 4 (Urban/Street):\n"
    "Detected visual relations: red bus driving on street; people walking on sidewalk; clock on tower\n"
    "Description: A red double-decker bus driving down a busy city street past pedestrians.\n\n"
)

def build_prompt_variants(img_entry):
    img_id = str(img_entry['id'])
    data = sgg_full_cache[img_id]
    triplets = data['triplets']
    objs = data['objects']
    
    raw_rel = format_triplets_raw(triplets, objs)
    filt_rel = format_triplets_filtered(triplets, objs)
    hier_rel = format_hierarchical(triplets, objs)
    
    prompts = {
        'P0_baseline_few': (
            SYS_STANDARD,
            DEMO_K2_BASELINE + f"Detected visual relations: {raw_rel}\nDescription:"
        ),
        'P1_filtered_triplets': (
            SYS_STANDARD,
            DEMO_K2_BASELINE + f"Detected visual relations: {filt_rel}\nDescription:"
        ),
        'P2_participle_syntax': (
            SYS_STANDARD,
            DEMO_PARTICIPLE + f"Detected visual relations: {raw_rel}\nDescription:"
        ),
        'P3_hierarchical_prompt': (
            SYS_STANDARD,
            (
                "EXAMPLE 1:\n"
                "Visual Evidence: Actions: dog chasing ball | Context: ball in grass; tree behind dog\n"
                "Description: A dog playfully chasing a tennis ball across a grassy park.\n\n"
                "EXAMPLE 2:\n"
                "Visual Evidence: Actions: person riding bike | Context: bike on sidewalk | Appearance: person wearing helmet\n"
                "Description: A person riding a bicycle down a sidewalk while wearing a helmet.\n\n"
                f"Visual Evidence: {hier_rel}\nDescription:"
            )
        ),
        'P4_diverse_k4': (
            SYS_STANDARD,
            DEMO_K4_DIVERSE + f"Detected visual relations: {raw_rel}\nDescription:"
        ),
        'P5_combined_optimal': (
            SYS_STANDARD,
            DEMO_K4_DIVERSE + f"Detected visual relations: {filt_rel}\nDescription:"
        )
    }
    return prompts

variant_names = [
    'P0_baseline_few',
    'P1_filtered_triplets',
    'P2_participle_syntax',
    'P3_hierarchical_prompt',
    'P4_diverse_k4',
    'P5_combined_optimal'
]

# Generate captions for all variants in batches
results_captions = {name: {} for name in variant_names}

BATCH_SIZE = 8
print(f"[{time.strftime('%X')}] Generating captions across {len(variant_names)} prompt strategies...")

for v_name in variant_names:
    t0 = time.time()
    for b_idx in range(0, len(test_images), BATCH_SIZE):
        batch = test_images[b_idx:b_idx + BATCH_SIZE]
        batch_prompts = []
        for entry in batch:
            sys_msg, user_msg = build_prompt_variants(entry)[v_name]
            msg = [{'role': 'system', 'content': sys_msg}, {'role': 'user', 'content': user_msg}]
            prompt = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            batch_prompts.append(prompt)
            
        inputs = tok(batch_prompts, return_tensors='pt', padding=True).to('cuda')
        prompt_len = inputs.input_ids.shape[1]
        with torch.no_grad():
            outputs = qwen.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
                pad_token_id=tok.eos_token_id
            )
            
        for i, entry in enumerate(batch):
            gen_tokens = outputs[i][prompt_len:]
            gen_text = tok.decode(gen_tokens, skip_special_tokens=True).strip()
            gen_text = gen_text.split('\n')[0].strip()
            results_captions[v_name][entry['id']] = gen_text
            
    print(f"  Finished {v_name} in {time.time() - t0:.1f}s", flush=True)

del qwen, tok
gc.collect()
torch.cuda.empty_cache()
print(f"[{time.strftime('%X')}] Generation complete. VRAM freed.")

# 4. Evaluation with pycocoevalcap
print(f"[{time.strftime('%X')}] Evaluating all {len(variant_names)} variants with pycocoevalcap...")
ref_path = SHARED / 'coco_karpathy/test_references_coco.json'
refs_data = json.loads(ref_path.read_text())

gts_raw = {}
target_set = set(test_ids)
for ann in refs_data['annotations']:
    if ann['image_id'] in target_set:
        gts_raw.setdefault(ann['image_id'], []).append({'caption': ann['caption']})

tokenizer = PTBTokenizer()
gts_tok = tokenizer.tokenize(gts_raw)

def evaluate_preds(preds_dict):
    preds_raw = {k: [{'caption': v}] for k, v in preds_dict.items()}
    res_tok = tokenizer.tokenize(preds_raw)
    scores = {}
    
    b_score, _ = Bleu(4).compute_score(gts_tok, res_tok)
    for i, s in enumerate(b_score, 1): scores[f'Bleu_{i}'] = float(s) * 100.0
    
    meteor_scorer = Meteor()
    try:
        m_score, _ = meteor_scorer.compute_score(gts_tok, res_tok)
        scores['METEOR'] = float(m_score) * 100.0
    finally:
        if hasattr(meteor_scorer, 'lock') and meteor_scorer.lock.locked():
            meteor_scorer.lock.release()
        del meteor_scorer
        
    r_score, _ = Rouge().compute_score(gts_tok, res_tok)
    scores['ROUGE_L'] = float(r_score) * 100.0
    
    c_score, _ = Cider().compute_score(gts_tok, res_tok)
    scores['CIDEr'] = float(c_score) * 100.0
    return scores

metrics_table = {}
for v_name in variant_names:
    s = evaluate_preds(results_captions[v_name])
    metrics_table[v_name] = s
    print(f"  {v_name:24s} -> CIDEr: {s['CIDEr']:.2f}, BLEU-4: {s['Bleu_4']:.2f}, METEOR: {s['METEOR']:.2f}, ROUGE-L: {s['ROUGE_L']:.2f}")

report = {
    'status': 'PROMPT_OPTIMIZATION_ABLATION_COMPLETE',
    'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    'wall_seconds': time.time() - START_TIME,
    'metrics': metrics_table,
    'samples': [
        {
            'image_id': entry['id'],
            'file_name': entry['file_name'],
            'triplets_raw': sgg_full_cache[str(entry['id'])]['triplets'][:6],
            'captions': {v_name: results_captions[v_name][entry['id']] for v_name in variant_names},
            'references': [a['caption'].strip() for a in gts_raw.get(entry['id'], [])]
        }
        for entry in test_images[:10]
    ]
}

out_path = ROOT / 'reports/COCO_PROMPT_OPTIMIZATION_ABLATION_RESULT.json'
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(report, indent=2))
print(f"Report saved to {out_path}")

print("SUMMARY_JSON_START")
print(json.dumps({'metrics': metrics_table, 'wall_seconds': round(report['wall_seconds'], 1)}, indent=2))
print("SUMMARY_JSON_END")

