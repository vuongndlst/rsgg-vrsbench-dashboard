from pathlib import Path
ROOT = Path.home() / 'rsg-former'
rep_file = ROOT / 'reports' / 'SYDNEY_CAPTIONS_ZEROSHOT_REPORT.json'
cache_file = ROOT / 'reports' / 'sydney_predictions_cache.json'
print("Sydney report exists:", rep_file.exists())
if rep_file.exists():
    print(rep_file.read_text(encoding='utf-8')[:500])
print("Sydney cache exists:", cache_file.exists())
if cache_file.exists():
    print("Cache size:", cache_file.stat().st_size)

