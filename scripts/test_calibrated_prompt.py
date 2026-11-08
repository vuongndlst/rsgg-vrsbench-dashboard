"""Test Length Calibration for Refined Prompting (9-13 words, max_new_tokens=28)"""
import os, sys, json, time, gc
from pathlib import Path
import torch

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
java_bin = os.path.expanduser('~/rsg-former/.tools/java/temurin8-52dcd578baca/jdk8u504-b01-jre/bin')
if os.path.exists(java_bin): os.environ['PATH'] = java_bin + ':' + os.environ['PATH']

ROOT = Path.home() / 'rsg-former'
SHARED = Path.home() / 'shared-assets/datasets'
sys.path.insert(0, str(ROOT / '.tools/deps'))
sys.path.insert(0, str(ROOT / 'src'))

from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider

manifest = json.loads((SHARED / 'coco_karpathy/test_manifest.json').read_text())
test_images = manifest['images'][:100]
sgg_cache = json.loads((ROOT / 'artifacts/coco_5000_sgg_native_cache.json').read_text())

from transformers import AutoTokenizer, AutoModelForCausalLM
model_path = '/home/stackops/etechs/msvd-tkg-captioning/models/qwen2.5-14b'
tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
if tok.pad_token_id is None: tok.pad_token_id = tok.eos_token_id
tok.padding_side = 'left'

qwen = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map='auto',
    local_files_only=True
)
qwen.eval()

sys_refined_calibrated = (
    "You are an expert image captioner. You are given detected visual relations from an image. "
    "Write exactly ONE natural English sentence (9 to 13 words) describing the main activity and setting. "
    "Do not describe clothing or body parts if an action or interaction is present. "
    "Do not speculate on weather, mood, or unseen items. Be direct, factual, and fluent."
)

demo_refined_calibrated = (
    "EXAMPLE 1:\n"
    "Detected visual relations: man riding motorcycle; motorcycle on dirt road; trees near road\n"
    "Description: A man is riding a motorcycle down a dirt road near trees.\n\n"
    "EXAMPLE 2:\n"
    "Detected visual relations: woman sitting at table; bowl of soup on table; woman holding fork\n"
    "Description: A woman is sitting at a wooden table eating from a bowl.\n\n"
    "EXAMPLE 3:\n"
    "Detected visual relations: dog chasing ball; dog running on grass; ball in grass\n"
    "Description: A dog is running across the green grass chasing after a ball.\n\n"
)

def format_rel(triplets, fallback_objs):
    if triplets:
        return '; '.join([f"{t['subject']} {t['predicate']} {t['object']}" for t in triplets[:8]])
    return ', '.join(fallback_objs[:5])

captions = {}
BATCH_SIZE = 8
for b_idx in range(0, len(test_images), BATCH_SIZE):
    batch = test_images[b_idx:b_idx + BATCH_SIZE]
    prompts = []
    for entry in batch:
        data = sgg_cache[str(entry['id'])]
        rel_str = format_rel(data['triplets'], data['objects'])
        user_msg = demo_refined_calibrated + f"Detected visual relations: {rel_str}\nDescription:"
        msg = [{'role': 'system', 'content': sys_refined_calibrated}, {'role': 'user', 'content': user_msg}]
        prompts.append(tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True))
        
    inputs = tok(prompts, return_tensors='pt', padding=True).to('cuda')
    prompt_len = inputs.input_ids.shape[1]
    with torch.no_grad():
        outputs = qwen.generate(**inputs, max_new_tokens=28, do_sample=False, pad_token_id=tok.eos_token_id)
        
    for i, entry in enumerate(batch):
        gen_tokens = outputs[i][prompt_len:]
        text = tok.decode(gen_tokens, skip_special_tokens=True).strip().split('\n')[0].strip()
        captions[entry['id']] = text

del qwen, tok
gc.collect()
torch.cuda.empty_cache()

refs_data = json.loads((SHARED / 'coco_karpathy/test_references_coco.json').read_text())
gts_raw = {}
for ann in refs_data['annotations']:
    if ann['image_id'] in captions:
        gts_raw.setdefault(ann['image_id'], []).append({'caption': ann['caption']})

tokenizer = PTBTokenizer()
gts_tok = tokenizer.tokenize(gts_raw)
res_tok = tokenizer.tokenize({k: [{'caption': v}] for k, v in captions.items()})

b_score, _ = Bleu(4).compute_score(gts_tok, res_tok)
meteor_scorer = Meteor()
try:
    m_score, _ = meteor_scorer.compute_score(gts_tok, res_tok)
finally:
    if hasattr(meteor_scorer, 'lock') and meteor_scorer.lock.locked(): meteor_scorer.lock.release()
    del meteor_scorer

r_score, _ = Rouge().compute_score(gts_tok, res_tok)
c_score, _ = Cider().compute_score(gts_tok, res_tok)

scores = {
    'CIDEr': float(c_score) * 100.0,
    'Bleu_4': float(b_score[3]) * 100.0,
    'Bleu_1': float(b_score[0]) * 100.0,
    'METEOR': float(m_score) * 100.0,
    'ROUGE_L': float(r_score) * 100.0
}
print("CALIBRATED_PROMPT_RESULTS:")
print(json.dumps(scores, indent=2))

