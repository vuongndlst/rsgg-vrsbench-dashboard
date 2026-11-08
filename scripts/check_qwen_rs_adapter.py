from pathlib import Path
p = Path.home() / 'rsg-former/artifacts/qwen_rs_lora'
print("qwen_rs_lora exists:", p.exists())
if p.exists():
    for f in p.iterdir():
        print(f"  {f.name} ({f.stat().st_size} bytes)")

