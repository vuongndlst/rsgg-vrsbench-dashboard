import json, os

cap_cache_path = os.path.join(os.path.expanduser("~"), "rsg-former/reports/VRSBENCH_DETECTOR_V2_FULL9350_CAPTIONS_CACHE.json")
if os.path.exists(cap_cache_path):
    data = json.load(open(cap_cache_path))
    print(f"VRSBench Full Captions generated so far: {len(data)} / 9350 ({len(data)/9350*100:.1f}%)")
else:
    print("VRSBench Full Captions cache not created yet (still in first 10 batches)")

