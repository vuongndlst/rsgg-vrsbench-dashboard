import json, os
from pathlib import Path

ROOT = Path.expanduser(Path('~/rsg-former'))

pred_path = ROOT / 'reports' / 'ucm_predictions_cache.json'
ds_path = ROOT / 'data' / 'ucm_captions' / 'dataset.json'

if pred_path.exists() and ds_path.exists():
    preds = json.loads(pred_path.read_text(encoding='utf-8'))
    data = json.loads(ds_path.read_text(encoding='utf-8'))
    
    img_by_id = {str(item['imgid']): item for item in data['images']}
    
    print(f"Loaded {len(preds)} predictions from ucm_predictions_cache.json")
    for qid in list(preds.keys())[:10]:
        item = img_by_id.get(qid, {})
        refs = [s['raw'] for s in item.get('sentences', [])]
        print(f"\n--- Image {item.get('filename', qid)} ---")
        print("PREDICTION  :", preds[qid][0])
        print("GROUND TRUTH:", refs[:2])

