import transformers, torch
print("transformers version:", transformers.__version__)
print("torch version:", torch.__version__)
from transformers import BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)
print("BitsAndBytesConfig created successfully!")

