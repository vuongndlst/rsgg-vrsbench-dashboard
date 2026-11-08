import os, sys, time
from pathlib import Path
ROOT = Path.home() / 'rsg-former'
sys.path.insert(0, str(ROOT))

print("Testing RSGGCaptionEngine loading...")
t0 = time.time()
try:
    from src.rsgg_caption_engine import RSGGCaptionEngine
    engine = RSGGCaptionEngine(device="cuda:0")
    print(f"Loaded successfully in {time.time()-t0:.2f}s!")
    cap = engine.generate_caption("sydney", [])
    print("Test caption generated:", cap)
except Exception as e:
    print("Engine test error:", e)
