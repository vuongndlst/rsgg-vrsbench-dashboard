import os
from pathlib import Path

ROOT = Path.expanduser(Path('~/rsg-former'))
art = ROOT / 'artifacts'
if art.exists():
    for p in sorted(art.glob('qwen*')):
        print(p.name, f"(dir={p.is_dir()})")
        if p.is_dir():
            for f in sorted(p.iterdir()):
                print(f"   {f.name} ({f.stat().st_size / 1024:.1f} KB)")

