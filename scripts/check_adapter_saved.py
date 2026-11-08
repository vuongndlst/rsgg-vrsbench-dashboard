from pathlib import Path
save_dir = Path.home() / 'rsg-former/artifacts/git_sgg_lora'
print("git_sgg_lora dir exists:", save_dir.exists())
if save_dir.exists():
    for p in save_dir.iterdir():
        print(f"  {p.name} ({p.stat().st_size} bytes)")

