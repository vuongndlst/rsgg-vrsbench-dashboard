import os, sys, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
model_path = '/home/stackops/etechs/msvd-tkg-captioning/models/qwen2.5-14b'

tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
if tok.pad_token_id is None:
    tok.pad_token_id = tok.eos_token_id
tok.padding_side = 'left'

qwen = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map='auto',
    local_files_only=True
)
qwen.eval()

prompts = [
    "<|im_start|>system\nYou are an image captioner.<|im_end|>\n<|im_start|>user\nDetected: dog playing with ball\nDescription:<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>system\nYou are an image captioner.<|im_end|>\n<|im_start|>user\nDetected: person holding umbrella on sidewalk; car on street\nDescription:<|im_end|>\n<|im_start|>assistant\n"
]

inputs = tok(prompts, return_tensors='pt', padding=True).to('cuda')
prompt_len = inputs.input_ids.shape[1]

with torch.no_grad():
    outputs = qwen.generate(**inputs, max_new_tokens=32, do_sample=False, pad_token_id=tok.eos_token_id)

print("Batch test outputs:")
for i in range(len(prompts)):
    gen_tokens = outputs[i][prompt_len:]
    text = tok.decode(gen_tokens, skip_special_tokens=True).strip()
    print(f"[{i}]: {text}")

print("BATCH_TEST_SUCCESS")

