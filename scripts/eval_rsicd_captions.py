"""Zero-Shot Captioning Evaluation on RSICD (Remote Sensing Image Captioning Dataset).
Evaluates the official 1,093-image test split:
  1. Frozen Detector v2 Best extracts bounding boxes and spatial relationships
  2. Scene graph passed to RSGGCaptionEngine (Qwen2.5-14B 4-bit) with RSICD profile
  3. Evaluates full metric suite: COCO (BLEU 1-4, METEOR, ROUGE-L, CIDEr) and SPICE (Java Stanford CoreNLP)
"""

import os, sys, json, time
from pathlib import Path
from typing import Dict, List, Any
from PIL import Image
import pandas as pd
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

def run_detector_inference(images_meta: List[Dict], img_dir: Path, device: torch.device) -> Dict[str, List[Dict]]:
    det_cache_path = ROOT / 'reports' / 'rsicd_detector_boxes_cache.json'
    if det_cache_path.exists():
        print(f"[Detector] Loading cached detections from {det_cache_path.name}...")
        try:
            raw = json.loads(det_cache_path.read_text(encoding='utf-8'))
            return raw
        except Exception as e:
            print(f"[Detector] Cache load error: {e}, recomputing...")

    print(f"[Detector] Running Detector v2 inference across {len(images_meta)} RSICD images...")
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
            img_id = str(item['imgid'])
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
            if (i + 1) % 200 == 0 or (i + 1) == len(images_meta):
                print(f"  [Detector] Processed {i+1}/{len(images_meta)} images ({time.time()-t0:.1f}s)...")

    det_cache_path.write_text(json.dumps(all_dets), encoding='utf-8')
    print(f"[Detector] Detections cached to {det_cache_path.name}")
    return all_dets

def main():
    print("========================================================")
    print("  RSICD ZERO-SHOT TRANSFER EVALUATION (1,093 TEST IMAGES)")
    print("========================================================")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Compute device: {device}")

    data_dir = ROOT / 'data' / 'rsicd'
    parquet_path = data_dir / 'test.parquet'
    img_dir = data_dir / 'images'

    if not parquet_path.exists():
        print(f"Error: {parquet_path} not found!", file=sys.stderr)
        return

    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df)} test rows from {parquet_path.name}")

    images_meta = []
    gts_test = {}
    for idx, row in df.iterrows():
        raw_fn = row.get('filename', f"rsicd_{idx:05d}.jpg")
        fn = os.path.basename(raw_fn)
        iid = str(idx)
        caps = [c.strip() for c in row.get('captions', []) if str(c).strip()]
        images_meta.append({'imgid': iid, 'filename': fn})
        gts_test[iid] = caps

    print(f"Prepared {len(images_meta)} test instances with {len(gts_test)} reference sets.")

    # 1. Run Detector v2
    detections_map = run_detector_inference(images_meta, img_dir, device)

    # 2. Generate Captions with RSGGCaptionEngine
    pred_cache_path = ROOT / 'reports' / 'rsicd_predictions_cache.json'
    res_test = {}
    if pred_cache_path.exists():
        print(f"Loading cached predictions from {pred_cache_path.name}...", flush=True)
        res_test = json.loads(pred_cache_path.read_text(encoding='utf-8'))
    else:
        print(f"[Caption] Generating Zero-Shot captions using tuned RSICD prompt with Qwen2.5-14B in batches...", flush=True)
        engine = RSGGCaptionEngine(device=str(device))
        t0 = time.time()
        test_qids = [item['imgid'] for item in images_meta]
        test_dets_list = [detections_map.get(qid, []) for qid in test_qids]

        caps = engine.generate_captions_batch("rsicd", test_dets_list, batch_size=8)
        for qid, cap in zip(test_qids, caps):
            res_test[qid] = [cap]

        pred_cache_path.write_text(json.dumps(res_test, indent=2), encoding='utf-8')
        print(f"Predictions saved to {pred_cache_path.name} ({time.time()-t0:.1f}s)", flush=True)

    # 3. Evaluate Metrics
    evaluator = MultiMetricEvaluator(enable_spice=True, enable_bertscore=False)
    
    print(f"\n--- EVALUATING RSICD OFFICIAL TEST SPLIT ({len(res_test)} images) ---")
    scores_test = evaluator.compute_all(gts_test, res_test)
    for k, v in sorted(scores_test.items()):
        print(f"  Test_{k}: {v:.2f}" if isinstance(v, float) else f"  Test_{k}: {v}")

    sota_baselines = {
        "SAT (Show Attend Tell)": {"BLEU-4": 38.2, "METEOR": 29.5, "ROUGE-L": 52.1, "CIDEr": 112.4},
        "MLCA-Net": {"BLEU-4": 42.6, "METEOR": 32.1, "ROUGE-L": 55.4, "CIDEr": 128.6},
        "RemoteCLIP-ZeroShot": {"BLEU-4": 28.5, "METEOR": 24.2, "ROUGE-L": 43.1, "CIDEr": 84.3}
    }

    report = {
        "dataset": "RSICD",
        "evaluation_type": "Zero-shot Cross-Domain Transfer (Trained only on VRSBench)",
        "detector_checkpoint": "vrsbench_detector_v2_best.pth (mIoU 70.35%)",
        "llm_backbone": "Qwen2.5-14B-Instruct (4-bit NF4)",
        "num_test_images": len(res_test),
        "metrics_test_split": scores_test,
        "comparison_sota": sota_baselines,
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
    }

    out_file = ROOT / 'reports' / 'RSICD_CAPTIONS_ZEROSHOT_REPORT.json'
    out_file.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f"\n[Summary] RSICD report written to {out_file.name}")

if __name__ == '__main__':
    main()

