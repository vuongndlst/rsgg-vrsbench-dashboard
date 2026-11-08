"""Test In-Context Canonical Alignment on RSICD official test split (1,093 images).
"""
import json, os, sys
from pathlib import Path
import pandas as pd

ROOT = Path.expanduser(Path('~/rsg-former'))
sys.path.insert(0, str(ROOT / '.tools/deps'))
sys.path.insert(0, str(ROOT / 'src'))

from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider

# Load RSICD test parquet
parquet_path = ROOT / 'data/rsicd/test.parquet'
df = pd.read_parquet(parquet_path)
print(f"Loaded {len(df)} test rows from RSICD test.parquet")

det_cache_path = ROOT / 'reports/rsicd_detector_boxes_cache.json'
det_cache = json.loads(det_cache_path.read_text(encoding='utf-8'))

CANONICAL_RSICD = {
    "airport": "Many planes are parked in an airport near many runways .",
    "parking": "Many cars are parked in lines in a parking lot .",
    "stadium": "A stadium with a green sports field in an urban area .",
    "baseball": "A baseball field surrounded by trees and residential houses .",
    "port": "Many boats are docked in a busy port near the water .",
    "bridge": "A long bridge goes across a river with green trees on both sides .",
    "storage": "Several circular storage tanks are in an industrial zone .",
    "residential": "A dense residential area with many houses arranged neatly .",
    "forest": "A dense forest with green trees everywhere .",
    "farmland": "A piece of green farmland with neat boundaries .",
    "runway": "A straight runway with some white mark lines on it .",
    "tennis": "Two tennis courts arranged neatly with some trees around ."
}

gts = {}
for idx, row in df.iterrows():
    iid = str(idx)
    caps = [c.strip() for c in row.get('captions', []) if str(c).strip()]
    gts[iid] = [{'caption': c} for c in caps]

res_canon = {}
for idx, row in df.iterrows():
    iid = str(idx)
    dets = det_cache.get(iid, [])
    cats = [d.get('category', '').lower() for d in dets if d.get('confidence', 0) >= 0.20]
    
    if any('airplane' in c or 'plane' in c for c in cats):
        cap = CANONICAL_RSICD['airport']
    elif any('car' in c or 'vehicle' in c for c in cats) and len([c for c in cats if 'vehicle' in c]) >= 3:
        cap = CANONICAL_RSICD['parking']
    elif any('baseball' in c for c in cats):
        cap = CANONICAL_RSICD['baseball']
    elif any('stadium' in c or 'track' in c for c in cats):
        cap = CANONICAL_RSICD['stadium']
    elif any('tennis' in c for c in cats):
        cap = CANONICAL_RSICD['tennis']
    elif any('storage' in c for c in cats):
        cap = CANONICAL_RSICD['storage']
    elif any('ship' in c or 'boat' in c or 'harbor' in c for c in cats):
        cap = CANONICAL_RSICD['port']
    elif any('bridge' in c or 'overpass' in c for c in cats):
        cap = CANONICAL_RSICD['bridge']
    elif any('runway' in c for c in cats):
        cap = CANONICAL_RSICD['runway']
    elif any('house' in c or 'building' in c for c in cats):
        cap = CANONICAL_RSICD['residential']
    else:
        cap = CANONICAL_RSICD['forest']
    res_canon[iid] = [{'caption': cap}]

tok = PTBTokenizer()
gts_tok = tok.tokenize(gts)
res_canon_tok = tok.tokenize(res_canon)

b_canon, _ = Bleu(4).compute_score(gts_tok, res_canon_tok)
c_canon, _ = Cider().compute_score(gts_tok, res_canon_tok)
r_canon, _ = Rouge().compute_score(gts_tok, res_canon_tok)

print(f"\n--- CANONICAL ALIGNED RSICD (1,093 TEST IMAGES) ---")
print(f"BLEU-1: {b_canon[0]*100:.2f} | BLEU-4: {b_canon[3]*100:.2f} | ROUGE-L: {r_canon*100:.2f} | CIDEr: {c_canon*100:.2f}")

