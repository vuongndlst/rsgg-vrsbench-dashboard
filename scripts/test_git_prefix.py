import os, torch
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
model_path = str(Path.home() / 'models/git-large-coco')

processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
git = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map='auto', local_files_only=True)
git.eval()

img_path = Path.home() / 'shared-assets/datasets/coco_karpathy/images/COCO_val2014_000000483108.jpg'
image = Image.open(img_path).convert('RGB')

# Test 1: Image only (standard)
inputs_img = processor(images=image, return_tensors='pt').to('cuda', dtype=torch.float16)
with torch.no_grad():
    out1 = git.generate(pixel_values=inputs_img.pixel_values, max_new_tokens=30)
print("Image only caption:", processor.decode(out1[0], skip_special_tokens=True))

# Test 2: Image + Prefix Text Prompt
prefix_text = "a man riding a bicycle"
inputs_prefix = processor(images=image, text=prefix_text, return_tensors='pt').to('cuda', dtype=torch.float16)
with torch.no_grad():
    out2 = git.generate(pixel_values=inputs_prefix.pixel_values, input_ids=inputs_prefix.input_ids, max_new_tokens=30)
print("Image + Prefix caption:", processor.decode(out2[0], skip_special_tokens=True))

