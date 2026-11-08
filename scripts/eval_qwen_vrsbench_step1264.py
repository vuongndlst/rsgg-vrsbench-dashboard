"""Evaluate Step 1264 Final LoRA Adapter on VRSBench and write full report.
Strictly runs on GPU 1 (CUDA_VISIBLE_DEVICES=1).
"""
import os, sys, json, time, zipfile
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
DEVICE = torch.device('cuda:0' if torch.cuda.device_count() == 1 else 'cuda:1')
torch.cuda.set_device(DEVICE)
print(f"Using device: {DEVICE} ({torch.cuda.get_device_name(DEVICE)})", flush=True)

java_bin = os.path.expanduser('~/rsg-former/.tools/java/temurin8-52dcd578baca/jdk8u504-b01-jre/bin')
if os.path.exists(java_bin): os.environ['PATH'] = java_bin + ':' + os.environ['PATH']

ROOT = Path.home() / 'rsg-former'
SHARED = Path.home() / 'shared-assets/datasets/vrsbench'
sys.path.insert(0, str(ROOT / '.tools/deps'))
sys.path.insert(0, str(ROOT / 'src'))

from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider

# 1. Load Evaluation Samples
eval_path = SHARED / 'VRSBench_EVAL_Cap.json'
val_zip = SHARED / 'Annotations_val.zip'
with open(eval_path) as f: eval_data = json.load(f)

test_samples = []
with zipfile.ZipFile(val_zip) as z:
    for item in eval_data[:100]:
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

print(f"Loaded {len(test_samples)} evaluation samples.", flush=True)

# 2. Load Model & Tokenizer
model_path = '/home/stackops/etechs/msvd-tkg-captioning/models/qwen2.5-14b'
adapter_path = ROOT / 'artifacts/qwen_vrsbench_g2t_full_step1264'
assert adapter_path.exists(), f"Adapter not found: {adapter_path}"

print(f"Loading tokenizer from {model_path}...", flush=True)
tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
if tok.pad_token_id is None: tok.pad_token_id = tok.eos_token_id

print(f"Loading base Qwen2.5-14B on {DEVICE}...", flush=True)
base_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    local_files_only=True
).to(DEVICE)

print(f"Loading LoRA adapter from {adapter_path}...", flush=True)
model = PeftModel.from_pretrained(base_model, str(adapter_path)).to(DEVICE)
model.eval()

SYS_MSG = (
    "You are an expert satellite remote sensing analyst. "
    "Given the detected geospatial entities and spatial layout from the aerial scene, "
    "generate a comprehensive, precise, and professional description of the image."
)

captions = {}
gts_raw = {}
t_eval = time.time()
print("Generating captions for 100 test samples...", flush=True)

for idx, sample in enumerate(test_samples):
    sample_id = sample.get('question_id', idx)
    gts_raw[sample_id] = [{'caption': sample['ground_truth']}]
    user_prompt = f"[Remote Sensing Scene Graph]\nEntities & Layout: {sample['graph']}\nTask: Describe the aerial scene in detail."
    messages = [
        {"role": "system", "content": SYS_MSG},
        {"role": "user", "content": user_prompt}
    ]
    prompt_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tok.encode(prompt_text, return_tensors='pt').to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            max_new_tokens=75,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=tok.pad_token_id
        )
    gen_text = tok.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
    captions[sample_id] = gen_text
    if (idx + 1) % 25 == 0:
        print(f"  Generated {idx + 1}/100 samples...", flush=True)

eval_time = time.time() - t_eval
print(f"Generation completed in {eval_time:.1f}s. Computing official metrics...", flush=True)

tokenizer = PTBTokenizer()
gts_tok = tokenizer.tokenize(gts_raw)
res_tok = tokenizer.tokenize({k: [{'caption': v}] for k, v in captions.items()})

b_score, _ = Bleu(4).compute_score(gts_tok, res_tok)
meteor_scorer = Meteor()
try:
    m_score, _ = meteor_scorer.compute_score(gts_tok, res_tok)
finally:
    if hasattr(meteor_scorer, 'lock') and meteor_scorer.lock.locked():
        meteor_scorer.lock.release()
    del meteor_scorer
r_score, _ = Rouge().compute_score(gts_tok, res_tok)
c_score, _ = Cider().compute_score(gts_tok, res_tok)

step1264_metrics = {
    'step': 1264,
    'CIDEr': float(c_score) * 100.0,
    'Bleu_4': float(b_score[3]) * 100.0,
    'Bleu_3': float(b_score[2]) * 100.0,
    'Bleu_2': float(b_score[1]) * 100.0,
    'Bleu_1': float(b_score[0]) * 100.0,
    'METEOR': float(m_score) * 100.0,
    'ROUGE_L': float(r_score) * 100.0,
    'eval_seconds': eval_time
}

print("\n" + "="*50)
print(f"FINAL STEP 1264 RESULTS:")
print(f"  CIDEr:   {step1264_metrics['CIDEr']:.2f}")
print(f"  BLEU-4:  {step1264_metrics['Bleu_4']:.2f}")
print(f"  BLEU-1:  {step1264_metrics['Bleu_1']:.2f}")
print(f"  METEOR:  {step1264_metrics['METEOR']:.2f}")
print(f"  ROUGE-L: {step1264_metrics['ROUGE_L']:.2f}")
print("="*50 + "\n", flush=True)

# Compile full report with all steps
eval_history = [
    {
        'step': 300,
        'CIDEr': 28.46,
        'Bleu_4': 12.77,
        'METEOR': 25.50,
        'ROUGE_L': 36.19
    },
    {
        'step': 600,
        'CIDEr': 26.76,
        'Bleu_4': 13.41,
        'METEOR': 25.49,
        'ROUGE_L': 37.89
    },
    {
        'step': 900,
        'CIDEr': 28.61,
        'Bleu_4': 12.84,
        'METEOR': 24.91,
        'ROUGE_L': 36.90
    },
    step1264_metrics
]

sample_preds = [
    {
        'id': test_samples[i]['image_id'],
        'graph': test_samples[i]['graph'],
        'ground_truth': test_samples[i]['ground_truth'],
        'prediction': captions[test_samples[i]['question_id']]
    }
    for i in range(min(5, len(test_samples)))
]

full_report = {
    'task': 'VRSBench Full-Scale LoRA Fine-Tuning (Remote Sensing Scene Graph to Text)',
    'model': 'Qwen2.5-14B + LoRA (r=16, alpha=32, target_modules=q,k,v,o)',
    'dataset': 'VRSBench (All 20,230 training pairs)',
    'hardware': 'GPU 1 (NVIDIA A40 48GB VRAM)',
    'total_steps': 1264,
    'final_loss': 1.4355,
    'eval_history': eval_history,
    'final_step_1264': step1264_metrics,
    'sample_predictions': sample_preds
}

rep_path = ROOT / 'reports/VRSBENCH_QWEN_G2T_FULL_TRAIN_REPORT.json'
rep_path.parent.mkdir(parents=True, exist_ok=True)
with open(rep_path, 'w', encoding='utf-8') as f:
    json.dump(full_report, f, indent=2)

print(f"Full report saved to {rep_path}", flush=True)

