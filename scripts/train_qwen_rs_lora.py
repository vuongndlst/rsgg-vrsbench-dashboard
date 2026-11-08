"""Fine-tune Qwen2.5-14B on Remote Sensing Geospatial Captioning (VRSBench benchmark).
Features:
- Fixed to GPU 1 (CUDA_VISIBLE_DEVICES=1)
- Trains LoRA adapter on 1,500 expert remote sensing captioning pairs from VRSBench
- Evaluates on 100 official test samples from VRSBench_EVAL_Cap.json
- Computes pycocoevalcap metrics: CIDEr, BLEU-1..4, METEOR, ROUGE-L
"""
import os, sys, json, time, gc, random
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
DEVICE = torch.device('cuda:0' if torch.cuda.device_count() == 1 else 'cuda:1')
torch.cuda.set_device(DEVICE)
print(f"Using device: {DEVICE} ({torch.cuda.get_device_name(DEVICE)})")

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

START_TIME = time.time()
print(f"[{time.strftime('%X')}] Starting Qwen2.5-14B Remote Sensing LoRA Fine-Tuning...")

# 1. Load VRSBench training data
vrs_train_path = SHARED / 'vrsbench/VRSBench_train.json'
assert vrs_train_path.exists(), f"Cannot find {vrs_train_path}"

with open(vrs_train_path) as f:
    vrs_data = json.load(f)

# Filter caption conversations
caption_entries = []
for entry in vrs_data:
    conv = entry.get('conversations', [])
    if len(conv) >= 2 and '[caption]' in conv[0].get('value', ''):
        human_q = conv[0]['value'].replace('<image>\n', '').replace('[caption]', '').strip()
        gpt_ans = conv[1]['value'].strip()
        if len(gpt_ans) > 20:
            caption_entries.append({
                'image': entry.get('image', ''),
                'prompt': human_q,
                'caption': gpt_ans
            })

random.seed(42)
random.shuffle(caption_entries)
train_pairs = caption_entries[:1500]
print(f"Selected {len(train_pairs)} remote sensing training pairs (from {len(caption_entries)} caption samples).")

# 2. Load Model & Tokenizer
model_path = '/home/stackops/etechs/msvd-tkg-captioning/models/qwen2.5-14b'
print(f"Loading Qwen2.5-14B from {model_path}...")
tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
if tok.pad_token_id is None: tok.pad_token_id = tok.eos_token_id

qwen = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    local_files_only=True
).to(DEVICE)
qwen.gradient_checkpointing_enable()
qwen.config.use_cache = False

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    lora_dropout=0.05,
    bias='none',
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(qwen, lora_config)
model.print_trainable_parameters()

# 3. Dataset & DataLoader
SYS_MSG = (
    "You are an expert satellite remote sensing analyst. "
    "Provide a detailed, precise, and professional description of the aerial scene."
)

class RSDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs
    def __len__(self):
        return len(self.pairs)
    def __getitem__(self, idx):
        item = self.pairs[idx]
        messages = [
            {"role": "system", "content": SYS_MSG},
            {"role": "user", "content": item['prompt']},
            {"role": "assistant", "content": item['caption']}
        ]
        prompt_text = tok.apply_chat_template(messages[:2], tokenize=False, add_generation_prompt=True)
        full_text = tok.apply_chat_template(messages, tokenize=False)
        
        prompt_ids = tok.encode(prompt_text, add_special_tokens=False)
        full_ids = tok.encode(full_text, add_special_tokens=False)
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
        
        # Max length truncation
        full_ids = full_ids[:256]
        labels = labels[:256]
        
        return {
            'input_ids': torch.tensor(full_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long)
        }

def collate_fn(batch):
    max_len = max(len(x['input_ids']) for x in batch)
    input_ids = []
    labels = []
    attention_mask = []
    pad_id = tok.pad_token_id
    for x in batch:
        pad_len = max_len - len(x['input_ids'])
        input_ids.append(torch.cat([x['input_ids'], torch.full((pad_len,), pad_id, dtype=torch.long)]))
        labels.append(torch.cat([x['labels'], torch.full((pad_len,), -100, dtype=torch.long)]))
        attention_mask.append(torch.cat([torch.ones(len(x['input_ids']), dtype=torch.long), torch.zeros(pad_len, dtype=torch.long)]))
    return {
        'input_ids': torch.stack(input_ids).to(DEVICE),
        'labels': torch.stack(labels).to(DEVICE),
        'attention_mask': torch.stack(attention_mask).to(DEVICE)
    }

train_loader = DataLoader(RSDataset(train_pairs), batch_size=4, shuffle=True, collate_fn=collate_fn)

# 4. Training Loop
EPOCHS = 1
GRAD_ACCUM = 2
total_steps = (len(train_loader) // GRAD_ACCUM) * EPOCHS
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.05), num_training_steps=total_steps)

print(f"[{time.strftime('%X')}] Training Qwen2.5-14B on Remote Sensing for {EPOCHS} epoch ({total_steps} steps)...")
model.train()
step = 0
epoch_start = time.time()

for epoch in range(EPOCHS):
    running_loss = 0.0
    for i, batch in enumerate(train_loader):
        outputs = model(**batch)
        loss = outputs.loss / GRAD_ACCUM
        loss.backward()
        running_loss += loss.item() * GRAD_ACCUM
        
        if (i + 1) % GRAD_ACCUM == 0 or (i + 1) == len(train_loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1
            
            if step % 25 == 0 or step == total_steps:
                lr = scheduler.get_last_lr()[0]
                print(f"  Step {step}/{total_steps} | Loss: {running_loss/25:.4f} | LR: {lr:.2e} | Elapsed: {time.time()-epoch_start:.1f}s")
                running_loss = 0.0

# Save adapter
save_dir = ROOT / 'artifacts/qwen_rs_lora'
save_dir.mkdir(parents=True, exist_ok=True)
model.save_pretrained(save_dir)
tok.save_pretrained(save_dir)
print(f"[{time.strftime('%X')}] Remote Sensing Qwen LoRA adapter saved to {save_dir}")

# 5. Evaluate on 100 Official VRSBench Evaluation Samples
print(f"[{time.strftime('%X')}] Evaluating on 100 VRSBench evaluation samples...")
eval_path = SHARED / 'vrsbench/VRSBench_EVAL_Cap.json'
with open(eval_path) as f:
    eval_data = json.load(f)

test_samples = eval_data[:100]
model.eval()
captions = {}
gts_raw = {}

for idx, sample in enumerate(test_samples):
    sample_id = sample.get('question_id', idx)
    gt_cap = sample.get('ground_truth', '')
    user_q = sample.get('question', 'Describe the image in detail')
    gts_raw[sample_id] = [{'caption': gt_cap}]
    
    messages = [
        {"role": "system", "content": SYS_MSG},
        {"role": "user", "content": user_q}
    ]
    prompt_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tok.encode(prompt_text, return_tensors='pt').to(DEVICE)
    
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            max_new_tokens=60,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=tok.pad_token_id
        )
    gen_text = tok.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
    captions[sample_id] = gen_text

del model, qwen, tok
gc.collect()
torch.cuda.empty_cache()

# Score with pycocoevalcap
print(f"[{time.strftime('%X')}] Computing pycocoevalcap metrics against Ground Truth...")
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

total_wall = time.time() - START_TIME
print("\n" + "="*50)
print("VRSBENCH REMOTE SENSING QWEN LoRA RESULTS (100 SAMPLES):")
print(f"  CIDEr   : {scores['CIDEr']:.2f}")
print(f"  BLEU-4  : {scores['Bleu_4']:.2f}")
print(f"  BLEU-1  : {scores['Bleu_1']:.2f}")
print(f"  METEOR  : {scores['METEOR']:.2f}")
print(f"  ROUGE-L : {scores['ROUGE_L']:.2f}")
print(f"  Wall Time: {total_wall:.1f}s")
print("="*50 + "\n")

report = {
    'status': 'VRSBENCH_QWEN_LORA_COMPLETE',
    'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    'wall_seconds': total_wall,
    'metrics': scores,
    'samples': [
        {
            'question_id': sample.get('question_id', idx),
            'generated': captions[sample.get('question_id', idx)],
            'ground_truth': sample.get('ground_truth', '')
        }
        for idx, sample in enumerate(test_samples[:10])
    ]
}

out_path = ROOT / 'reports/VRSBENCH_QWEN_LORA_PILOT100_RESULT.json'
out_path.write_text(json.dumps(report, indent=2))
print(f"Report saved to {out_path}")
