import json
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from peft import PeftModel

ROOT = Path.home() / 'rsg-former'
SHARED = Path.home() / 'shared-assets/datasets/coco_karpathy'
DEVICE = torch.device('cuda:1')

model_path = str(Path.home() / 'models/git-large-coco')
processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
base = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, local_files_only=True).to(DEVICE)
adapter_dir = str(ROOT / 'artifacts/git_sgg_lora')
model = PeftModel.from_pretrained(base, adapter_dir).to(DEVICE).eval()

manifest = json.loads((SHARED / 'test_manifest.json').read_text())
test_images = manifest['images'][:5]
sgg_cache = json.loads((ROOT / 'artifacts/coco_5000_sgg_native_cache.json').read_text())

print("Testing generation with repetition penalty & beam search:")
for entry in test_images:
    img_path = SHARED / 'images' / entry['file_name']
    image = Image.open(img_path).convert('RGB')
    data = sgg_cache[str(entry['id'])]
    triplets = data['triplets']
    objs = data['objects']
    rel_str = '; '.join([f"{t['subject']} {t['predicate']} {t['object']}" for t in triplets[:8]]) if triplets else ', '.join(objs[:5])
    prefix = f"visual relations: {rel_str} description: "
    
    inputs = processor(images=image, text=prefix, return_tensors='pt').to(DEVICE, dtype=torch.float16)
    with torch.no_grad():
        out = model.generate(
            pixel_values=inputs.pixel_values,
            input_ids=inputs.input_ids,
            max_new_tokens=25,
            repetition_penalty=2.0,
            no_repeat_ngram_size=3,
            num_beams=3
        )
    prompt_len = inputs.input_ids.shape[1]
    gen_text = processor.decode(out[0][prompt_len:], skip_special_tokens=True).strip()
    print(f"ID {entry['id']}: {gen_text}")

