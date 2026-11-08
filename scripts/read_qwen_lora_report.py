from pathlib import Path
p = Path.home() / 'rsg-former/reports/COCO_QWEN_LORA_G2T_PILOT100_RESULT.json'
print(p.read_text())

