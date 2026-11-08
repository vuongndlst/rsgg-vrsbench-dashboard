"""Compare official pycocoevalcap scores: Batch vs Single on 50 test samples.
Runs on GPU 0.
"""
import os, sys, time, json, zipfile
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
DEVICE = torch.device('cuda:0')

ROOT = Path.home() / 'rsg-former'
SHARED = Path.home() / 'shared-assets/datasets/vrsbench'
sys.path.insert(0, str(ROOT / '.tools/deps'))
sys.path.insert(0, str(ROOT / 'src'))

java_bin = os.path.expanduser('~/rsg-former/.tools/java/temurin8-52dcd578baca/jdk8u504-b01-jre/bin')
if os.path.exists(java_bin): os.environ['PATH'] = java_bin + ':' + os.environ['PATH']

from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider

eval_path = SHARED / 'VRSBench_EVAL_Cap.json'
val_zip = SHARED / 'Annotations_val.zip'
with open(eval_path) as f: eval_data = json.load(f)

test_samples = []
with zipfile.ZipFile(val_zip) as z:
    for item in eval_data[:50]:
        img_id = item.get('image_id', '')
        base_name = img_id.replace('.png', '').replace('.jpg', '')
        cand = f"Annotations_val/{base_name}.json"
        data = json.loads(z.read(cand).decode('utf-8'))
        objs = data.get('objects', [])
        obj_summary = []
        for o in objs:
            cls_name = o.get('obj_cls', 'object')
            size = o.get('obj_size', '')
            ref = o.get('referring_sentence', '')
            item_desc = f"{size + ' ' if size else ''}{cls_name}"
            if ref: item_desc += f" ({ref.strip('.')})"
            obj_summary.append(item_desc)
        test_samples.append({
            'question_id': item.get('question_id'),
            'image_id': img_id,
            'graph': "; ".join(obj_summary) if obj_summary else "aerial scene overview",
            'ground_truth': item.get('ground_truth', data.get('caption', ''))
        })

print(f"Loaded {len(test_samples)} samples.", flush=True)

model_path = '/home/stackops/etechs/msvd-tkg-captioning/models/qwen2.5-14b'
adapter_path = ROOT / 'artifacts/qwen_vrsbench_g2t_full_step900'

tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
if tok.pad_token_id is None: tok.pad_token_id = tok.eos_token_id
tok.padding_side = 'left'

base_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, local_files_only=True).to(DEVICE)
model = PeftModel.from_pretrained(base_model, str(adapter_path)).to(DEVICE)
model.eval()

SYS_MSG = (
    "You are an expert satellite remote sensing analyst. "
    "Given the detected geospatial entities and spatial layout from the aerial scene, "
    "generate a comprehensive, precise, and professional description of the image."
)

# 1. Batched inference (batch_size=10)
print("\n--- Running Batched Inference (B=10) on 50 samples ---", flush=True)
t0 = time.time()
batch_captions = {}
B = 10
for b_idx in range(0, len(test_samples), B):
    batch = test_samples[b_idx:b_idx+B]
    prompts = []
    for s in batch:
        user_prompt = f"[Remote Sensing Scene Graph]\nEntities & Layout: {s['graph']}\nTask: Describe the aerial scene in detail."
        msg = [{"role": "system", "content": SYS_MSG}, {"role": "user", "content": user_prompt}]
        prompts.append(tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True))
    
    inputs = tok(prompts, return_tensors='pt', padding=True).to(DEVICE)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=75, do_sample=False, repetition_penalty=1.1, pad_token_id=tok.pad_token_id)
    prompt_len = inputs.input_ids.shape[1]
    for i, s in enumerate(batch):
        gen = tok.decode(out[i][prompt_len:], skip_special_tokens=True).strip()
        batch_captions[s['question_id']] = gen

t_batch = time.time() - t0
print(f"Batched generation took {t_batch:.1f}s ({t_batch/50:.2f}s/sample)", flush=True)

# 2. Single inference (batch_size=1)
print("\n--- Running Single Inference (B=1) on 50 samples ---", flush=True)
t0 = time.time()
single_captions = {}
for s in test_samples:
    user_prompt = f"[Remote Sensing Scene Graph]\nEntities & Layout: {s['graph']}\nTask: Describe the aerial scene in detail."
    msg = [{"role": "system", "content": SYS_MSG}, {"role": "user", "content": user_prompt}]
    prompt_text = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    input_ids = tok.encode(prompt_text, return_tensors='pt').to(DEVICE)
    with torch.no_grad():
        out = model.generate(input_ids=input_ids, max_new_tokens=75, do_sample=False, repetition_penalty=1.1, pad_token_id=tok.pad_token_id)
    gen = tok.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
    single_captions[s['question_id']] = gen

t_single = time.time() - t0
print(f"Single generation took {t_single:.1f}s ({t_single/50:.2f}s/sample)", flush=True)

# 3. Compute Metrics for Both
gts_raw = {s['question_id']: [{'caption': s['ground_truth']}] for s in test_samples}
tokenizer = PTBTokenizer()
gts_tok = tokenizer.tokenize(gts_raw)

def score(caps):
    res_tok = tokenizer.tokenize({k: [{'caption': v}] for k, v in caps.items()})
    b, _ = Bleu(4).compute_score(gts_tok, res_tok)
    m_scorer = Meteor()
    try: m, _ = m_scorer.compute_score(gts_tok, res_tok)
    finally:
        if hasattr(m_scorer, 'lock') and m_scorer.lock.locked(): m_scorer.lock.release()
        del m_scorer
    r, _ = Rouge().compute_score(gts_tok, res_tok)
    c, _ = Cider().compute_score(gts_tok, res_tok)
    return {'CIDEr': c*100, 'BLEU-4': b[3]*100, 'METEOR': m*100, 'ROUGE-L': r*100}

score_batch = score(batch_captions)
score_single = score(single_captions)

print("\n" + "="*50)
print("SCORE COMPARISON (50 SAMPLES):")
print(f"  Batch  (B=10): CIDEr={score_batch['CIDEr']:.2f}, BLEU-4={score_batch['BLEU-4']:.2f}, METEOR={score_batch['METEOR']:.2f}, ROUGE-L={score_batch['ROUGE-L']:.2f} (Time: {t_batch:.1f}s)")
print(f"  Single (B=1):  CIDEr={score_single['CIDEr']:.2f}, BLEU-4={score_single['BLEU-4']:.2f}, METEOR={score_single['METEOR']:.2f}, ROUGE-L={score_single['ROUGE-L']:.2f} (Time: {t_single:.1f}s)")
print("="*50, flush=True)

