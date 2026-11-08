import json
from pathlib import Path

rep_dir = Path.home() / 'rsg-former' / 'reports'
f1 = rep_dir / 'VRSBENCH_QWEN_G2T_FULL9350_EVAL_REPORT.json'
if f1.exists():
    print("=== VRSBENCH_QWEN_G2T_FULL9350_EVAL_REPORT.json ===")
    print(f1.read_text(encoding='utf-8'))

f2 = rep_dir / 'VRSBENCH_QWEN_G2T_ORACLE_CAPTIONS_CACHE.json'
if f2.exists():
    print(f"=== VRSBENCH_QWEN_G2T_ORACLE_CAPTIONS_CACHE.json ({f2.stat().st_size} bytes) ===")
    with open(f2) as f:
        d = json.load(f)
    print(f"Total entries in Oracle cache: {len(d)}")
    print("Sample key:", list(d.keys())[0])

