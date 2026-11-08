import json, os

cap_cache_path = os.path.join(os.path.expanduser("~"), "rsg-former/reports/VRSBENCH_DETECTOR_V2_FULL9350_CAPTIONS_CACHE.json")
eval_json_path = os.path.join(os.path.expanduser("~"), "shared-assets/datasets/vrsbench/VRSBench_EVAL_Cap.json")

preds = json.load(open(cap_cache_path))
eval_items = json.load(open(eval_json_path))

print(f"Total predictions: {len(preds)}, Total eval items: {len(eval_items)}")
eval_by_id = {str(it['question_id']): it for it in eval_items}

for qid in list(preds.keys())[:5]:
    gt = eval_by_id.get(qid, {}).get('ground_truth', '')
    pred = preds[qid]
    print(f"\n--- QID {qid} ---")
    print(f"PRED: {pred}")
    print(f"GT:   {gt[:150]}...")

