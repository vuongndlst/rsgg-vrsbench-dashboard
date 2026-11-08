"""Fine-tune Qwen2.5-14B LoRA on Remote Sensing Scene Graph to Text (RSSG -> Text) on VRSBench.
Features:
- Fixed strictly to GPU 1 (CUDA_VISIBLE_DEVICES=1)
- Trains LoRA adapter on 2,000 Remote Sensing Scene Graph -> Expert Caption pairs from VRSBench
- Evaluates on 100 official test samples from VRSBench_EVAL_Cap.json with their respective RSSG
- Computes pycocoevalcap metrics: CIDEr, BLEU-1..4, METEOR, ROUGE-L
- Saves report to reports/VRSBENCH_QWEN_G2T_LORA_PILOT100_RESULT.json
"""
import os, sys, json, time, gc, random, zipfile
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
SHARED = Path.home() / 'shared-assets/datasets/vrsbench'
sys.path.insert(0, str(ROOT / '.tools/deps'))
sys.path.insert(0, str(ROOT / 'src'))

from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider

START_TIME = time.time()
print(f"[{time.strftime('%X')}] Starting Qwen2.5-14B Remote Sensing Graph-to-Text (G2T) LoRA Fine-Tuning...")

# 1. Load VRSBench training data with Scene Graphs
train_zip = SHARED / 'Annotations_train.zip'
assert train_zip.exists(), f"Cannot find {train_zip}"

train_pairs = []
with zipfile.ZipFile(train_zip) as z:
    file_list = [f for f in z.namelist() if f.endswith('.json') and not f.startswith('__MACOSX')]
    random.seed(42)
    random.shuffle(file_list)
    for fn in file_list:
        if len(train_pairs) >= 2000: break
        data = json.loads(z.read(fn).decode('utf-8'))
        objs = data.get('objects', [])
        obj_summary = []
        for o in objs:
            cls_name = o.get('obj_cls', 'object')
            size = o.get('obj_size', '')
            ref = o.get('referring_sentence', '')
            item = f"{size + ' ' if size else ''}{cls_name}"
            if ref:
                item += f" ({ref.strip('.')})"
            obj_summary.append(item)
        cap = data.get('caption', '')
        if obj_summary and len(cap) > 20:
            train_pairs.append({
                'id': fn,
                'graph': "; ".join(obj_summary),
                'caption': cap
            })

print(f"Loaded {len(train_pairs)} training pairs with Remote Sensing Scene Graphs.")

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
    "Given the detected geospatial entities and spatial layout from the aerial scene, "
    "generate a comprehensive, precise, and professional description of the image."
)

class RSG2TDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs
    def __len__(self):
        return len(self.pairs)
    def __getitem__(self, idx):
        item = self.pairs[idx]
        user_prompt = f"[Remote Sensing Scene Graph]\nEntities & Layout: {item['graph']}\nTask: Describe the aerial scene in detail."
        messages = [
            {"role": "system", "content": SYS_MSG},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": item['caption']}
        ]
        prompt_text = tok.apply_chat_template(messages[:2], tokenize=False, add_generation_prompt=True)
        full_text = tok.apply_chat_template(messages, tokenize=False)
        
        prompt_ids = tok.encode(prompt_text, add_special_tokens=False)
        full_ids = tok.encode(full_text, add_special_tokens=False)
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
        
        full_ids = full_ids[:320]
        labels = labels[:320]
        
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

train_loader = DataLoader(RSG2TDataset(train_pairs), batch_size=4, shuffle=True, collate_fn=collate_fn)

# 4. Training Loop
EPOCHS = 1
GRAD_ACCUM = 2
total_steps = (len(train_loader) // GRAD_ACCUM) * EPOCHS
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.05), num_training_steps=total_steps)

print(f"[{time.strftime('%X')}] Training Qwen2.5-14B on Remote Sensing Graph-to-Text for {EPOCHS} epoch ({total_steps} steps)...", flush=True)
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
                print(f"  Step {step}/{total_steps} | Loss: {running_loss/25:.4f} | LR: {lr:.2e} | Elapsed: {time.time()-epoch_start:.1f}s", flush=True)
                running_loss = 0.0

# Save adapter
save_dir = ROOT / 'artifacts/qwen_vrsbench_g2t_lora'
save_dir.mkdir(parents=True, exist_ok=True)
model.save_pretrained(save_dir)
tok.save_pretrained(save_dir)
print(f"[{time.strftime('%X')}] Remote Sensing G2T LoRA adapter saved to {save_dir}", flush=True)

# 5. Evaluate on 100 Official VRSBench Evaluation Samples with Ground-Truth RSSG
print(f"[{time.strftime('%X')}] Evaluating on 100 VRSBench evaluation samples using Scene Graphs...", flush=True)
eval_path = SHARED / 'VRSBench_EVAL_Cap.json'
val_zip = SHARED / 'Annotations_val.zip'

with open(eval_path) as f:
    eval_data = json.load(f)

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
            if ref:
                item_desc += f" ({ref.strip('.')})"
            obj_summary.append(item_desc)
        test_samples.append({
            'question_id': item.get('question_id'),
            'image_id': img_id,
            'graph': "; ".join(obj_summary) if obj_summary else "aerial scene overview",
            'ground_truth': item.get('ground_truth', data.get('caption', ''))
        })

model.eval()
qwen.config.use_cache = True  # Enable KV cache for fast inference!
captions = {}
gts_raw = {}

gen_start = time.time()
for idx, sample in enumerate(test_samples):
    sample_id = sample.get('question_id', idx)
    gt_cap = sample.get('ground_truth', '')
    gts_raw[sample_id] = [{'caption': gt_cap}]
    
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
    if (idx + 1) % 20 == 0:
        print(f"  Generated {idx+1}/100 samples in {time.time()-gen_start:.1f}s", flush=True)

del model, qwen, tok
gc.collect()
torch.cuda.empty_cache()

# Score with pycocoevalcap
print(f"[{time.strftime('%X')}] Computing pycocoevalcap metrics against Ground Truth...", flush=True)
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
print("\n" + "="*50, flush=True)
print("VRSBENCH REMOTE SENSING GRAPH-TO-TEXT (G2T) RESULTS (100 SAMPLES):", flush=True)
print(f"  CIDEr   : {scores['CIDEr']:.2f}", flush=True)
print(f"  BLEU-4  : {scores['Bleu_4']:.2f}", flush=True)
print(f"  BLEU-1  : {scores['Bleu_1']:.2f}", flush=True)
print(f"  METEOR  : {scores['METEOR']:.2f}", flush=True)
print(f"  ROUGE-L : {scores['ROUGE_L']:.2f}", flush=True)
print(f"  Wall Time: {total_wall:.1f}s", flush=True)
print("="*50 + "\n", flush=True)

report = {
    'status': 'VRSBENCH_QWEN_G2T_LORA_COMPLETE',
    'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    'wall_seconds': total_wall,
    'metrics': scores,
    'samples': [
        {
            'question_id': sample['question_id'],
            'image_id': sample['image_id'],
            'graph': sample['graph'],
            'generated': captions[sample['question_id']],
            'ground_truth': sample['ground_truth']
        }
        for sample in test_samples[:10]
    ]
}

out_path = ROOT / 'reports/VRSBENCH_QWEN_G2T_LORA_PILOT100_RESULT.json'
out_path.write_text(json.dumps(report, indent=2))
print(f"Report saved to {out_path}", flush=True)

