from pathlib import Path
p = Path.home() / "EduSGG/scripts/train_qwen_qlora.py"
if p.exists():
    text = p.read_text(encoding="utf-8")
    for i, line in enumerate(text.splitlines()):
        if any(k in line for k in ["from_pretrained", "quant", "bnb", "BitsAndBytesConfig", "torch_dtype", "device_map"]):
            print(f"{i}: {line}")
