import json
from pathlib import Path

lora_dir = Path.home() / "rsg-former" / "artifacts" / "qwen_vrsbench_g2t_lora"
cfg_path = lora_dir / "adapter_config.json"
readme_path = lora_dir / "README.md"

if cfg_path.exists():
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    print("Old base_model_name_or_path in config:", cfg.get("base_model_name_or_path"))
    cfg["base_model_name_or_path"] = "Qwen/Qwen2.5-14B-Instruct"
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print("Updated base_model_name_or_path to Qwen/Qwen2.5-14B-Instruct in adapter_config.json")

if readme_path.exists():
    text = readme_path.read_text(encoding="utf-8")
    print("Found README.md in lora folder. Length:", len(text))
    # Replace any local path with Qwen/Qwen2.5-14B-Instruct
    lines = []
    for line in text.splitlines():
        if "base_model" in line and "/" in line:
            lines.append("base_model: Qwen/Qwen2.5-14B-Instruct")
        else:
            lines.append(line)
    readme_path.write_text("\n".join(lines), encoding="utf-8")
    print("Updated README.md in lora folder")

