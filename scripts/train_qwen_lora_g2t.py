"""Fine-tune Qwen2.5-14B with LoRA on Graph-to-Text COCO captions.
Trains on 2,515 (RSG-Former SGG Triplets -> Ground-Truth Caption) pairs from val split.
Evaluates on 100 COCO Karpathy test images to measure the BLEU-4 surge.
"""
import os, sys, json, time, gc, random
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType

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

START_TIME = time.time()
print(f"[{time.strftime('%X')}] Starting Qwen-14B LoRA Fine-Tuning for Graph-to-Text...")

# 1. Load training dataset
train_data_path = ROOT / 'artifacts/coco_val500_sgg_train_dataset.json'
assert train_data_path.exists(), "Training dataset not found!"
raw_train_records = json.loads(train_data_path.read_text())

# Flatten into (rel_str, caption) pairs
train_pairs = []
for item in raw_train_records:
    rel_str = item['rel_str']
    for cap in item['captions']:
        train_pairs.append({'rel_str': rel_str, 'caption': cap.strip()})

random.seed(42)
random.shuffle(train_pairs)
print(f"Loaded {len(train_pairs)} training pairs from {len(raw_train_records)} images.")

# 2. Load Tokenizer & Base Model
model_path = '/home/stackops/etechs/msvd-tkg-captioning/models/qwen2.5-14b'
tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
if tok.pad_token_id is None: tok.pad_token_id = tok.eos_token_id

qwen = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map='auto',
    local_files_only=True
)
qwen.gradient_checkpointing_enable()

# Configure LoRA
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

# 3. Prepare Training Batches
SYS_PROMPT = (
    "You are an expert image captioner. You are given detected visual relations from an image. "
    "Write exactly ONE natural English sentence (9 to 13 words) describing the main activity and setting."
)

class G2TDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs
    def __len__(self):
        return len(self.pairs)
    def __getitem__(self, idx):
        item = self.pairs[idx]
        user_msg = f"Detected visual relations: {item['rel_str']}\nDescription:"
        prompt_msgs = [{'role': 'system', 'content': SYS_PROMPT}, {'role': 'user', 'content': user_msg}]
        prompt_text = tok.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
        full_text = prompt_text + item['caption'] + tok.eos_token
        
        prompt_ids = tok(prompt_text, add_special_tokens=False).input_ids
        full_ids = tok(full_text, add_special_tokens=False).input_ids
        
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
        return {
            'input_ids': torch.tensor(full_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long)
        }

def collate_fn(batch):
    max_len = max(len(x['input_ids']) for x in batch)
    input_ids = []
    labels = []
    attention_mask = []
    for x in batch:
        pad_len = max_len - len(x['input_ids'])
        # Right pad for causal training
        input_ids.append(torch.cat([x['input_ids'], torch.full((pad_len,), tok.pad_token_id, dtype=torch.long)]))
        labels.append(torch.cat([x['labels'], torch.full((pad_len,), -100, dtype=torch.long)]))
        attention_mask.append(torch.cat([torch.ones(len(x['input_ids']), dtype=torch.long), torch.zeros(pad_len, dtype=torch.long)]))
    return {
        'input_ids': torch.stack(input_ids).cuda(),
        'labels': torch.stack(labels).cuda(),
        'attention_mask': torch.stack(attention_mask).cuda()
    }

train_loader = DataLoader(G2TDataset(train_pairs), batch_size=4, shuffle=True, collate_fn=collate_fn)

# 4. Training Loop
EPOCHS = 2
GRAD_ACCUM = 2
total_steps = (len(train_loader) // GRAD_ACCUM) * EPOCHS
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.05), num_training_steps=total_steps)

print(f"[{time.strftime('%X')}] Training for {EPOCHS} epochs ({total_steps} optimizer steps)...")
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
            
            if step % 50 == 0 or step == total_steps:
                lr = scheduler.get_last_lr()[0]
                print(f"  Step {step}/{total_steps} | Loss: {running_loss/50:.4f} | LR: {lr:.2e} | Elapsed: {time.time()-epoch_start:.1f}s")
                running_loss = 0.0

# Save LoRA adapter
save_dir = ROOT / 'artifacts/qwen_coco_lora'
save_dir.mkdir(parents=True, exist_ok=True)
model.save_pretrained(save_dir)
print(f"[{time.strftime('%X')}] LoRA adapter saved to {save_dir}")

# 5. Evaluate on 100 Pilot Images
print(f"[{time.strftime('%X')}] Evaluating fine-tuned Qwen LoRA on 100 COCO Karpathy images...")
manifest = json.loads((SHARED / 'coco_karpathy/test_manifest.json').read_text())
test_images = manifest['images'][:100]
sgg_cache = json.loads((ROOT / 'artifacts/coco_5000_sgg_native_cache.json').read_text())

model.eval()
tok.padding_side = 'left'
captions = {}
BATCH_SIZE = 8

for b_idx in range(0, len(test_images), BATCH_SIZE):
    batch = test_images[b_idx:b_idx + BATCH_SIZE]
    prompts = []
    for entry in batch:
        data = sgg_cache[str(entry['id'])]
        triplets = data['triplets']
        objs = data['objects']
        rel_str = '; '.join([f"{t['subject']} {t['predicate']} {t['object']}" for t in triplets[:8]]) if triplets else ', '.join(objs[:5])
        user_msg = f"Detected visual relations: {rel_str}\nDescription:"
        msg = [{'role': 'system', 'content': SYS_PROMPT}, {'role': 'user', 'content': user_msg}]
        prompts.append(tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True))
        
    inputs = tok(prompts, return_tensors='pt', padding=True).to('cuda')
    prompt_len = inputs.input_ids.shape[1]
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=28, do_sample=False, pad_token_id=tok.eos_token_id)
        
    for i, entry in enumerate(batch):
        gen_tokens = outputs[i][prompt_len:]
        text = tok.decode(gen_tokens, skip_special_tokens=True).strip().split('\n')[0].strip()
        captions[entry['id']] = text

del model, qwen, tok
gc.collect()
torch.cuda.empty_cache()

# Score with pycocoevalcap
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

total_wall = time.time() - START_TIME
print("\n" + "="*50)
print("QWEN-14B LoRA (GRAPH-TO-TEXT) BENCHMARK RESULTS (100 PILOT):")
print(f"  CIDEr   : {scores['CIDEr']:.2f}")
print(f"  BLEU-4  : {scores['Bleu_4']:.2f}")
print(f"  BLEU-1  : {scores['Bleu_1']:.2f}")
print(f"  METEOR  : {scores['METEOR']:.2f}")
print(f"  ROUGE-L : {scores['ROUGE_L']:.2f}")
print(f"  Wall Time: {total_wall:.1f}s")
print("="*50 + "\n")

report = {
    'status': 'QWEN_LORA_G2T_COMPLETE',
    'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    'wall_seconds': total_wall,
    'metrics': scores,
    'samples': [
        {
            'image_id': entry['id'],
            'caption': captions[entry['id']],
            'ground_truth': [a['caption'] for a in gts_raw.get(entry['id'], [])]
        }
        for entry in test_images[:10]
    ]
}

out_path = ROOT / 'reports/COCO_QWEN_LORA_G2T_PILOT100_RESULT.json'
out_path.write_text(json.dumps(report, indent=2))
print(f"Report saved to {out_path}")

print("SUMMARY_JSON_START")
print(json.dumps({'metrics': scores, 'wall_seconds': round(total_wall, 1)}, indent=2))
print("SUMMARY_JSON_END")

