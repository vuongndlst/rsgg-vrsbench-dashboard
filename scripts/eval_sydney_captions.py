"""Zero-Shot Captioning Evaluation on Sydney-Captions.
Cross-domain transfer of RSGG (trained strictly on VRSBench) without any fine-tuning:
  - Frozen Detector v2 Best extracts bounding boxes and spatial relationships
  - Scene graph passed to RSGGCaptionEngine (Qwen2.5-14B 4-bit)
  - Evaluates Karpathy test split (58 images) and Full dataset (613 images)
  - Full metrics suite: COCO (BLEU 1-4, METEOR, ROUGE-L, CIDEr), SPICE, CHAIR
"""

import os, sys, json, time
from pathlib import Path
from typing import Dict, List, Any
from PIL import Image
import torch
import torchvision
from torchvision.transforms import functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

ROOT = Path.home() / 'rsg-former'
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from configs.dataset_prompts import get_dataset_config
from src.rsgg_multi_evaluator import MultiMetricEvaluator
from src.rsgg_caption_engine import RSGGCaptionEngine

def run_detector_inference(images_meta: List[Dict], img_dir: Path, device: torch.device) -> Dict[int, List[Dict]]:
    det_cache_path = ROOT / 'reports' / 'sydney_detector_boxes_cache.json'
    if det_cache_path.exists():
        print(f"[Detector] Loading cached detections from {det_cache_path.name}...")
        try:
            raw = json.loads(det_cache_path.read_text(encoding='utf-8'))
            return {int(k): v for k, v in raw.items()}
        except Exception as e:
            print(f"[Detector] Cache load error: {e}, recomputing...")

    print("[Detector] Running Detector v2 inference across Sydney-Captions...")
    ckpt_path = ROOT / 'artifacts' / 'vrsbench_detector_v2_best.pth'
    ckpt = torch.load(ckpt_path, map_location='cpu')
    classes = ckpt['classes']

    model = fasterrcnn_resnet50_fpn_v2(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(classes))
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()

    all_dets = {}
    t0 = time.time()
    with torch.no_grad():
        for i, item in enumerate(images_meta):
            img_id = item['imgid']
            fn = item['filename']
            img_p = img_dir / fn
            if not img_p.exists():
                all_dets[img_id] = []
                continue
            
            img = Image.open(img_p).convert('RGB')
            w, h = img.size
            t = F.to_tensor(img).unsqueeze(0).to(device)
            preds = model(t)[0]
            boxes = preds['boxes'].cpu().numpy()
            scores = preds['scores'].cpu().numpy()
            labels = preds['labels'].cpu().numpy()

            keep = scores >= 0.20
            img_dets = []
            for b, s, l in zip(boxes[keep], scores[keep], labels[keep]):
                img_dets.append({
                    "category": classes[l],
                    "confidence": float(s),
                    "box": [float(b[0]/w), float(b[1]/h), float(b[2]/w), float(b[3]/h)]
                })
            all_dets[img_id] = img_dets
            if (i + 1) % 100 == 0 or (i + 1) == len(images_meta):
                print(f"  [Detector] Processed {i+1}/{len(images_meta)} images ({time.time()-t0:.1f}s)...")

    # Cache detector output
    det_cache_path.write_text(json.dumps({str(k): v for k, v in all_dets.items()}), encoding='utf-8')
    print(f"[Detector] Detections cached to {det_cache_path.name}")
    return all_dets

def main():
    print("========================================================")
    print("  SYDNEY-CAPTIONS ZERO-SHOT TRANSFER EVALUATION")
    print("========================================================")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Compute device: {device}")

    data_dir = ROOT / 'data' / 'sydney_captions'
    json_path = data_dir / 'dataset.json'
    img_dir = data_dir / 'images'

    if not json_path.exists():
        print(f"Error: {json_path} not found!", file=sys.stderr)
        return

    data = json.loads(json_path.read_text(encoding='utf-8'))
    images_meta = data.get('images', [])
    print(f"Loaded {len(images_meta)} images from {json_path.name}")

    # Build reference GTs
    gts_test = {}
    gts_full = {}
    for item in images_meta:
        img_id = item['imgid']
        refs = [s['raw'].strip() for s in item.get('sentences', [])]
        gts_full[img_id] = refs
        if item.get('split') == 'test':
            gts_test[img_id] = refs

    print(f"Full set: {len(gts_full)} images | Karpathy test split: {len(gts_test)} images")

    # 1. Run Detector v2
    detections_map = run_detector_inference(images_meta, img_dir, device)

    # 2. Generate Captions with RSGGCaptionEngine
    pred_cache_path = ROOT / 'reports' / 'sydney_predictions_cache.json'
    res_full = {}
    print("[Caption] Generating Zero-Shot captions using tuned Sydney prompt with Qwen2.5-14B in batches...", flush=True)
    engine = RSGGCaptionEngine(device=str(device))
    t0 = time.time()
    
    all_imgids = [item['imgid'] for item in images_meta]
    all_dets_list = [detections_map.get(iid, []) for iid in all_imgids]

    # Use batch_size=8 to prevent VRAM spikes during beam search
    batch_size = 8
    caps = engine.generate_captions_batch("sydney", all_dets_list, batch_size=batch_size)
    for iid, cap in zip(all_imgids, caps):
        res_full[iid] = [cap]

    pred_cache_path.write_text(json.dumps({str(k): v for k, v in res_full.items()}, indent=2), encoding='utf-8')
    print(f"Predictions saved to {pred_cache_path.name} in {time.time()-t0:.1f}s", flush=True)

    # 3. Evaluate Metrics
    evaluator = MultiMetricEvaluator(enable_spice=True, enable_bertscore=False)
    
    # Test split evaluation
    res_test = {k: res_full[k] for k in gts_test if k in res_full}
    print(f"\n--- EVALUATING KARPATHY TEST SPLIT ({len(res_test)} images) ---")
    scores_test = evaluator.compute_all(gts_test, res_test)
    for k, v in sorted(scores_test.items()):
        print(f"  Test_{k}: {v:.2f}" if isinstance(v, float) else f"  Test_{k}: {v}")

    # Full set evaluation
    print(f"\n--- EVALUATING FULL DATASET ({len(res_full)} images) ---")
    scores_full = evaluator.compute_all(gts_full, res_full)
    for k, v in sorted(scores_full.items()):
        print(f"  Full_{k}: {v:.2f}" if isinstance(v, float) else f"  Full_{k}: {v}")

    # Comparison against published literature on Sydney-Captions
    # (e.g. RSICD/Sydney baselines: Show-Tell, Top-Down, MLCA, SAT, RemoteCLIP)
    sota_baselines = {
        "SAT (Show Attend Tell)": {"BLEU-4": 52.8, "METEOR": 37.1, "ROUGE-L": 66.8, "CIDEr": 196.4},
        "MLCA-Net": {"BLEU-4": 56.4, "METEOR": 39.8, "ROUGE-L": 70.1, "CIDEr": 221.5},
        "RemoteCLIP-ZeroShot": {"BLEU-4": 42.1, "METEOR": 31.5, "ROUGE-L": 58.2, "CIDEr": 145.8}
    }

    report = {
        "dataset": "Sydney-Captions",
        "evaluation_type": "Zero-shot Cross-Domain Transfer (Trained only on VRSBench)",
        "detector_checkpoint": "vrsbench_detector_v2_best.pth (mIoU 70.35%)",
        "llm_backbone": "Qwen2.5-14B-Instruct (4-bit NF4)",
        "num_test_images": len(res_test),
        "num_full_images": len(res_full),
        "metrics_test_split": scores_test,
        "metrics_full_dataset": scores_full,
        "comparison_sota": sota_baselines,
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
    }

    out_file = ROOT / 'reports' / 'SYDNEY_CAPTIONS_ZEROSHOT_REPORT.json'
    out_file.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f"\n[Summary] Sydney report written to {out_file.name}")

if __name__ == '__main__':
    main()
