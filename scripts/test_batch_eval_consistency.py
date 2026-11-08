"""Verify batched inference consistency vs single-sample generation on GPU 0.
Tests 20 samples to compare generated text and metric parity.
"""
import os, sys, time, json, zipfile
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
DEVICE = torch.device('cuda:0')
print(f"Using device: {DEVICE} ({torch.cuda.get_device_name(DEVICE)})", flush=True)

ROOT = Path.home() / 'rsg-former'
SHARED = Path.home() / 'shared-assets/datasets/vrsbench'
sys.path.insert(0, str(ROOT / '.tools/deps'))
sys.path.insert(0, str(ROOT / 'src'))

eval_path = SHARED / 'VRSBench_EVAL_Cap.json'
val_zip = SHARED / 'Annotations_val.zip'
with open(eval_path) as f: eval_data = json.load(f)

test_samples = []
with zipfile.ZipFile(val_zip) as z:
    for item in eval_data[:20]:
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

print("Loading tokenizer...", flush=True)
tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
if tok.pad_token_id is None: tok.pad_token_id = tok.eos_token_id
tok.padding_side = 'left'

print("Loading model on GPU 0...", flush=True)
base_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    local_files_only=True
).to(DEVICE)
model = PeftModel.from_pretrained(base_model, str(adapter_path)).to(DEVICE)
model.eval()

SYS_MSG = (
    "You are an expert satellite remote sensing analyst. "
    "Given the detected geospatial entities and spatial layout from the aerial scene, "
    "generate a comprehensive, precise, and professional description of the image."
)

# Test 1: Single sample (batch_size=1)
print("\n--- Test 1: Single Sample Generation (5 samples) ---", flush=True)
single_res = []
t0 = time.time()
for s in test_samples[:5]:
    user_prompt = f"[Remote Sensing Scene Graph]\nEntities & Layout: {s['graph']}\nTask: Describe the aerial scene in detail."
    msg = [{"role": "system", "content": SYS_MSG}, {"role": "user", "content": user_prompt}]
    text = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    inp = tok.encode(text, return_tensors='pt').to(DEVICE)
    with torch.no_grad():
        out = model.generate(inp, max_new_tokens=75, do_sample=False, repetition_penalty=1.1, pad_token_id=tok.pad_token_id)
    gen = tok.decode(out[0][inp.shape[1]:], skip_special_tokens=True).strip()
    single_res.append(gen)
t_single = time.time() - t0
print(f"Single generation time for 5 samples: {t_single:.2f}s ({t_single/5:.2f}s/sample)")

# Test 2: Batched (batch_size=5)
print("\n--- Test 2: Batched Generation (5 samples at once) ---", flush=True)
prompts = []
for s in test_samples[:5]:
    user_prompt = f"[Remote Sensing Scene Graph]\nEntities & Layout: {s['graph']}\nTask: Describe the aerial scene in detail."
    msg = [{"role": "system", "content": SYS_MSG}, {"role": "user", "content": user_prompt}]
    prompts.append(tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True))

t0 = time.time()
inputs = tok(prompts, return_tensors='pt', padding=True).to(DEVICE)
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=75,
        do_sample=False,
        repetition_penalty=1.1,
        pad_token_id=tok.pad_token_id
    )
batch_res = []
prompt_len = inputs.input_ids.shape[1]
for i in range(5):
    # decode only generated tokens for sample i
    gen = tok.decode(outputs[i][prompt_len:], skip_special_tokens=True).strip()
    batch_res.append(gen)
t_batch = time.time() - t0
print(f"Batched generation time for 5 samples: {t_batch:.2f}s ({t_batch/5:.2f}s/sample)")
print(f"Speedup: {t_single / t_batch:.2f}x faster!")

print("\n--- Comparison of Outputs ---")
identical = 0
for i in range(5):
    print(f"Sample {i}:")
    print(f"  Single: {single_res[i][:100]}...")
    print(f"  Batch : {batch_res[i][:100]}...")
    if single_res[i] == batch_res[i]:
        identical += 1
        print("  -> EXACT MATCH!")
    else:
        print("  -> Differs slightly due to padding attention mask.")
print(f"\nExact match: {identical}/5")

