"""Full-scale fine-tuning of Qwen2.5-14B LoRA on the entire VRSBench dataset (20,230 samples).
Architecture: Remote Sensing Scene Graph (RSSG) -> Holistic Expert Caption.
Features:
- Fixed strictly to GPU 1 (CUDA_VISIBLE_DEVICES=1)
- Trains on all 20,230 remote sensing scene graph pairs from VRSBench
- Effective batch size = 16 (batch_size=4, grad_accum=4)
- Cosine LR schedule with warmup
- Intermediate evaluations & checkpoints at steps 300, 600, 900, and final
- Evaluates on 100 official VRSBench evaluation samples with ground-truth RSSG
- Measures CIDEr, BLEU-1..4, METEOR, ROUGE-L via pycocoevalcap
- Outputs comprehensive trajectory to reports/VRSBENCH_QWEN_G2T_FULL_TRAIN_REPORT.json
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

START_TIME = time.time()
print(f"[{time.strftime('%X')}] Starting FULL-SCALE Qwen2.5-14B Remote Sensing Graph-to-Text LoRA Training...", flush=True)

# 1. Load entire VRSBench training dataset
train_zip = SHARED / 'Annotations_train.zip'
assert train_zip.exists(), f"Cannot find {train_zip}"

train_pairs = []
with zipfile.ZipFile(train_zip) as z:
    file_list = [f for f in z.namelist() if f.endswith('.json') and not f.startswith('__MACOSX')]
    random.seed(42)
    random.shuffle(file_list)
    for fn in file_list:
        data = json.loads(z.read(fn).decode('utf-8'))
        objs = data.get('objects', [])
        obj_summary = []
        for o in objs:
            cls_name = o.get('obj_cls', 'object')
            size = o.get('obj_size', '')
            ref = o.get('referring_sentence', '')
            item = f"{size + ' ' if size else ''}{cls_name}"
            if ref: item += f" ({ref.strip('.')})"
            obj_summary.append(item)
        cap = data.get('caption', '')
        if obj_summary and len(cap) > 20:
            train_pairs.append({
                'id': fn,
                'graph': "; ".join(obj_summary),
                'caption': cap
            })

print(f"Loaded ALL {len(train_pairs)} training pairs with Remote Sensing Scene Graphs.", flush=True)

# 2. Load Model & Tokenizer
model_path = '/home/stackops/etechs/msvd-tkg-captioning/models/qwen2.5-14b'
print(f"Loading Qwen2.5-14B from {model_path}...", flush=True)
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

class FullRSDataset(Dataset):
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

train_loader = DataLoader(FullRSDataset(train_pairs), batch_size=4, shuffle=True, collate_fn=collate_fn)

# 4. Load Evaluation Samples (100 official samples)
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

def run_evaluation(current_step):
    print(f"[{time.strftime('%X')}] --- Running Evaluation at Step {current_step} ---", flush=True)
    model.eval()
    qwen.config.use_cache = True
    captions = {}
    gts_raw = {}
    t_eval = time.time()
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
    
    tokenizer = PTBTokenizer()
    gts_tok = tokenizer.tokenize(gts_raw)
    res_tok = tokenizer.tokenize({k: [{'caption': v}] for k, v in captions.items()})
    b_score, _ = Bleu(4).compute_score(gts_tok, res_tok)
    meteor_scorer = Meteor()
    try: m_score, _ = meteor_scorer.compute_score(gts_tok, res_tok)
    finally:
        if hasattr(meteor_scorer, 'lock') and meteor_scorer.lock.locked(): meteor_scorer.lock.release()
        del meteor_scorer
    r_score, _ = Rouge().compute_score(gts_tok, res_tok)
    c_score, _ = Cider().compute_score(gts_tok, res_tok)
    
    metrics = {
        'step': current_step,
        'CIDEr': float(c_score) * 100.0,
        'Bleu_4': float(b_score[3]) * 100.0,
        'Bleu_1': float(b_score[0]) * 100.0,
        'METEOR': float(m_score) * 100.0,
        'ROUGE_L': float(r_score) * 100.0,
        'eval_seconds': time.time() - t_eval
    }
    print(f"  Step {current_step} | CIDEr: {metrics['CIDEr']:.2f} | BLEU-4: {metrics['Bleu_4']:.2f} | METEOR: {metrics['METEOR']:.2f} | ROUGE-L: {metrics['ROUGE_L']:.2f} ({metrics['eval_seconds']:.1f}s)", flush=True)
    model.train()
    qwen.config.use_cache = False
    return metrics, captions

# 5. Training Loop Setup
EPOCHS = 1
GRAD_ACCUM = 4
total_steps = (len(train_loader) // GRAD_ACCUM) * EPOCHS
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.05), num_training_steps=total_steps)

print(f"[{time.strftime('%X')}] Commencing FULL training on {len(train_pairs)} pairs ({total_steps} steps, batch_size=4, grad_accum=4)...", flush=True)
model.train()
step = 0
epoch_start = time.time()
running_loss = 0.0
eval_trajectory = []
checkpoints_steps = {300, 600, 900, total_steps}

for epoch in range(EPOCHS):
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
                elapsed = time.time() - epoch_start
                rate = elapsed / step
                eta = (total_steps - step) * rate
                print(f"  Step {step}/{total_steps} | Loss: {running_loss/50:.4f} | LR: {lr:.2e} | Elapsed: {elapsed/60:.1f}m | ETA: {eta/60:.1f}m", flush=True)
                running_loss = 0.0
            
            # Periodic evaluation & saving
            if step in checkpoints_steps:
                save_dir = ROOT / f"artifacts/qwen_vrsbench_g2t_full_step{step}"
                save_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(save_dir)
                tok.save_pretrained(save_dir)
                print(f"  [Checkpoint] Saved adapter to {save_dir}", flush=True)
                
                metrics, caps = run_evaluation(step)
                eval_trajectory.append({
                    'step': step,
                    'metrics': metrics,
                    'sample_output': caps.get(test_samples[0]['question_id'], '')
                })

# Final save
final_dir = ROOT / 'artifacts/qwen_vrsbench_g2t_full_final'
final_dir.mkdir(parents=True, exist_ok=True)
model.save_pretrained(final_dir)
tok.save_pretrained(final_dir)
print(f"[{time.strftime('%X')}] Final adapter saved to {final_dir}", flush=True)

report = {
    'status': 'VRSBENCH_FULL_TRAINING_COMPLETE',
    'total_training_samples': len(train_pairs),
    'total_steps': total_steps,
    'wall_seconds': time.time() - START_TIME,
    'trajectory': eval_trajectory
}

out_path = ROOT / 'reports/VRSBENCH_QWEN_G2T_FULL_TRAIN_REPORT.json'
out_path.write_text(json.dumps(report, indent=2))
print(f"Full training report saved to {out_path}", flush=True)

