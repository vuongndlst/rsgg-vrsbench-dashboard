from pathlib import Path
lora_dir = Path.home() / "rsg-former" / "artifacts" / "qwen_vrsbench_g2t_lora"
readme = lora_dir / "README.md"
if readme.exists():
    readme.unlink()
    print("Removed inner README.md from lora folder so root README.md serves as primary Model Card")

