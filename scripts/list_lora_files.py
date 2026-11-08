from pathlib import Path
lora_dir = Path.home() / "rsg-former" / "artifacts" / "qwen_vrsbench_g2t_lora"
for p in lora_dir.iterdir():
    print(p.name, p.stat().st_size)

