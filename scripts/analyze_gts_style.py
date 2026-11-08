import json, os
from pathlib import Path
from collections import Counter

ROOT = Path.expanduser(Path('~/rsg-former'))

def analyze_gts(dataset_name, json_path):
    if not os.path.exists(json_path):
        print(f"File not found: {json_path}")
        return
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"\n=== {dataset_name} ({len(data)} samples) ===")
    all_words = []
    stems = []
    lengths = []
    sample_caps = []
    for item in list(data.values())[:30]:
        caps = item if isinstance(item, list) else [item]
        for c in caps:
            words = c.strip().lower().split()
            lengths.append(len(words))
            if len(words) >= 3:
                stems.append(" ".join(words[:3]))
            all_words.extend(words)
            if len(sample_caps) < 5:
                sample_caps.append(c)
    print(f"Avg length: {sum(lengths)/len(lengths):.1f} words (range {min(lengths)} - {max(lengths)})")
    print("Top 3-gram sentence openers:", Counter(stems).most_common(6))
    print("Sample ground truths:")
    for sc in sample_caps:
        print("  -", sc)

analyze_gts("UCM", ROOT / "data/ucm/ucm_captions_gt.json")
analyze_gts("Sydney", ROOT / "data/sydney/sydney_captions_gt.json")

