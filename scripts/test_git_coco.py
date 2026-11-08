"""Evaluate Microsoft GIT-Large-COCO on the same 100 COCO Karpathy pilot images.
Computes official COCO metrics (Bleu-1..4, METEOR, ROUGE-L, CIDEr) to establish
the end-to-end supervised benchmark ceiling.
"""
import os, sys, json, time, gc
from pathlib import Path
import torch
from PIL import Image

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

from transformers import AutoProcessor, AutoModelForCausalLM

model_path = str(Path.home() / 'models/git-large-coco')
print(f"Loading GIT-Large-COCO from {model_path}...")
processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
git_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map='auto',
    local_files_only=True
)
git_model.eval()
print("GIT-Large-COCO loaded.")

captions = {}
t0 = time.time()
print("Generating captions for 100 pilot images with GIT-Large-COCO...")
for idx, entry in enumerate(test_images):
    img_path = SHARED / 'coco_karpathy/images' / entry['file_name']
    image = Image.open(img_path).convert('RGB')
    inputs = processor(images=image, return_tensors='pt').to('cuda', dtype=torch.float16)
    with torch.no_grad():
        out = git_model.generate(pixel_values=inputs.pixel_values, max_new_tokens=30)
    caption = processor.decode(out[0], skip_special_tokens=True).strip()
    captions[entry['id']] = caption
    if (idx + 1) % 25 == 0:
        print(f"  Processed {idx + 1}/100 images...")

print(f"Generation done in {time.time() - t0:.1f}s.")

del git_model, processor
gc.collect()
torch.cuda.empty_cache()

# Evaluate
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
print("GIT_LARGE_COCO_RESULTS:")
print(json.dumps(scores, indent=2))

samples = []
for entry in test_images[:5]:
    samples.append({
        'id': entry['id'],
        'caption': captions[entry['id']],
        'gt': [a['caption'] for a in gts_raw.get(entry['id'], [])]
    })
print("Sample Comparisons:")
print(json.dumps(samples, indent=2))

