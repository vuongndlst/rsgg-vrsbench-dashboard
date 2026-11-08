import os, glob

base = os.path.join(os.path.expanduser("~"), "rsg-former")
rep = os.path.join(base, "reports")

target_files = [
    "VRSBENCH_DETECTOR_V2_SINGLE_BOXES_9350.json",
    "VRSBENCH_DETECTOR_V2_SG_FULL9350_CACHE.json",
    "VRSBENCH_DETECTOR_V2_FULL9350_CAPTIONS_CACHE.json",
    "VRSBENCH_FULL9350_DETECTOR_V2_SG_CAPTION_REPORT.json",
    "ucm_detector_boxes_cache.json",
    "ucm_predictions_cache.json",
    "UCM_CAPTIONS_ZEROSHOT_REPORT.json",
    "sydney_detector_boxes_cache.json",
    "sydney_predictions_cache.json",
    "SYDNEY_CAPTIONS_ZEROSHOT_REPORT.json",
    "DIOR_RSVG_ZEROSHOT_REPORT.json"
]

print("=== CHECKING TARGET REPORTS IN RSG-FORMER ===")
for tf in target_files:
    p = os.path.join(rep, tf)
    if os.path.exists(p):
        sz = os.path.getsize(p)
        print(f"FOUND: {tf} ({sz} bytes, {sz/(1024*1024):.2f} MB)")
    else:
        print(f"MISSING: {tf}")

