"""Compute comprehensive Object Detection & Visual Grounding metrics (mIoU, mAP@50, Acc@0.7, Acc@0.5, mAP@50:95)
for the trained Faster R-CNN (ResNet-50-FPN, Epoch 5) on the 1,000 VRSBench test images.
"""
import json
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0

def normalize_cls(cls_name):
    c = (cls_name or "").strip().lower()
    for p in ["small ", "large ", "small-", "large-"]:
        if c.startswith(p):
            c = c[len(p):]
    return c

def compute_ap_all_point(recalls, precisions):
    """Standard VOC2010+ / COCO all-point interpolated Average Precision."""
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return float(ap)

def compute_ap_101_point(recalls, precisions):
    """COCO 101-point interpolated Average Precision."""
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        inds = np.where(mrec >= t)[0]
        p = np.max(mpre[inds]) if inds.size > 0 else 0.0
        ap += p / 101.0
    return float(ap)

def evaluate_ap_for_threshold(samples, iou_thresh=0.5, class_aware=True):
    """Compute AP at a given IoU threshold."""
    if not class_aware:
        # Single micro/overall pool
        gt_by_img = {}
        all_preds = []
        total_gt = 0
        for s in samples:
            qid = str(s["question_id"])
            gts = [g["obj_coord"] for g in s.get("objects", []) if len(g.get("obj_coord", [])) == 4]
            gt_by_img[qid] = {"boxes": gts, "matched": [False] * len(gts)}
            total_gt += len(gts)
            for p in s.get("detected_objects", []):
                if len(p.get("obj_coord", [])) == 4:
                    all_preds.append((float(p.get("score", 0.5)), qid, p["obj_coord"]))
        all_preds.sort(key=lambda x: x[0], reverse=True)
        tp = np.zeros(len(all_preds))
        fp = np.zeros(len(all_preds))
        for i, (sc, qid, pbox) in enumerate(all_preds):
            ginfo = gt_by_img[qid]
            best_iou = 0.0
            best_j = -1
            for j, gbox in enumerate(ginfo["boxes"]):
                iou = compute_iou(pbox, gbox)
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            if best_iou >= iou_thresh and best_j >= 0 and not ginfo["matched"][best_j]:
                tp[i] = 1.0
                ginfo["matched"][best_j] = True
            else:
                fp[i] = 1.0
        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        recalls = cum_tp / max(1, total_gt)
        precisions = cum_tp / np.maximum(1e-9, cum_tp + cum_fp)
        return {
            "ap_all_point": compute_ap_all_point(recalls, precisions) * 100.0,
            "ap_101_point": compute_ap_101_point(recalls, precisions) * 100.0,
            "max_recall": float(recalls[-1] * 100.0) if len(recalls) else 0.0,
            "final_precision": float(precisions[-1] * 100.0) if len(precisions) else 0.0,
            "tp": int(np.sum(tp)),
            "fp": int(np.sum(fp)),
            "total_gt": total_gt
        }

    # Class-aware AP per category
    classes = set()
    for s in samples:
        for g in s.get("objects", []):
            if len(g.get("obj_coord", [])) == 4:
                classes.add(normalize_cls(g.get("obj_cls", "")))
    per_class = {}
    for cls in sorted(classes):
        if not cls:
            continue
        gt_by_img = {}
        all_preds = []
        total_gt = 0
        for s in samples:
            qid = str(s["question_id"])
            gts = [
                g["obj_coord"]
                for g in s.get("objects", [])
                if len(g.get("obj_coord", [])) == 4 and normalize_cls(g.get("obj_cls", "")) == cls
            ]
            gt_by_img[qid] = {"boxes": gts, "matched": [False] * len(gts)}
            total_gt += len(gts)
            for p in s.get("detected_objects", []):
                if len(p.get("obj_coord", [])) == 4 and normalize_cls(p.get("obj_cls", "")) == cls:
                    all_preds.append((float(p.get("score", 0.5)), qid, p["obj_coord"]))
        if total_gt == 0:
            continue
        all_preds.sort(key=lambda x: x[0], reverse=True)
        tp = np.zeros(len(all_preds))
        fp = np.zeros(len(all_preds))
        for i, (sc, qid, pbox) in enumerate(all_preds):
            ginfo = gt_by_img[qid]
            best_iou = 0.0
            best_j = -1
            for j, gbox in enumerate(ginfo["boxes"]):
                iou = compute_iou(pbox, gbox)
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            if best_iou >= iou_thresh and best_j >= 0 and not ginfo["matched"][best_j]:
                tp[i] = 1.0
                ginfo["matched"][best_j] = True
            else:
                fp[i] = 1.0
        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        recalls = cum_tp / max(1, total_gt)
        precisions = cum_tp / np.maximum(1e-9, cum_tp + cum_fp)
        per_class[cls] = {
            "ap_all_point": compute_ap_all_point(recalls, precisions) * 100.0,
            "ap_101_point": compute_ap_101_point(recalls, precisions) * 100.0,
            "recall": float(recalls[-1] * 100.0) if len(recalls) else 0.0,
            "precision": float(precisions[-1] * 100.0) if len(precisions) else 0.0,
            "gt_count": total_gt,
            "pred_count": len(all_preds),
            "tp": int(np.sum(tp))
        }
    macro_ap = float(np.mean([v["ap_all_point"] for v in per_class.values()])) if per_class else 0.0
    weighted_ap = (
        float(
            np.sum([v["ap_all_point"] * v["gt_count"] for v in per_class.values()])
            / max(1, sum(v["gt_count"] for v in per_class.values()))
        )
        if per_class
        else 0.0
    )
    return {
        "macro_mAP": round(macro_ap, 2),
        "weighted_mAP": round(weighted_ap, 2),
        "per_class": per_class
    }

def main():
    bundle_path = ROOT / "reports/dashboard_bundle.json"
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    samples = bundle["samples"]

    # 1. Compute IoU, Acc@0.5, Acc@0.7 (Both Class-Agnostic & Class-Aware, Best-Overlap & 1-to-1 Greedy Matching)
    ious_best_agnostic = []
    ious_best_class = []
    ious_1to1_agnostic = []
    ious_1to1_class = []
    ious_detected_only = []

    for s in samples:
        gts = [(normalize_cls(g.get("obj_cls", "")), g["obj_coord"]) for g in s.get("objects", []) if len(g.get("obj_coord", [])) == 4]
        preds = [
            (normalize_cls(p.get("obj_cls", "")), p["obj_coord"], float(p.get("score", 0.5)))
            for p in s.get("detected_objects", [])
            if len(p.get("obj_coord", [])) == 4
        ]
        preds.sort(key=lambda x: x[2], reverse=True)

        # Best overlap per GT box
        for gcls, gbox in gts:
            best_ag = max([compute_iou(gbox, pbox) for _, pbox, _ in preds], default=0.0)
            best_cl = max([compute_iou(gbox, pbox) for pcls, pbox, _ in preds if pcls == gcls], default=0.0)
            ious_best_agnostic.append(best_ag)
            ious_best_class.append(best_cl)
            if best_ag > 0.0:
                ious_detected_only.append(best_ag)

        # 1-to-1 greedy matching (sorted by IoU descending)
        used_p_ag = set()
        for gcls, gbox in gts:
            best_j = -1
            best_val = 0.0
            for j, (_, pbox, _) in enumerate(preds):
                if j in used_p_ag:
                    continue
                val = compute_iou(gbox, pbox)
                if val > best_val:
                    best_val = val
                    best_j = j
            if best_j >= 0 and best_val > 0:
                used_p_ag.add(best_j)
                ious_1to1_agnostic.append(best_val)
            else:
                ious_1to1_agnostic.append(0.0)

        used_p_cl = set()
        for gcls, gbox in gts:
            best_j = -1
            best_val = 0.0
            for j, (pcls, pbox, _) in enumerate(preds):
                if j in used_p_cl or pcls != gcls:
                    continue
                val = compute_iou(gbox, pbox)
                if val > best_val:
                    best_val = val
                    best_j = j
            if best_j >= 0 and best_val > 0:
                used_p_cl.add(best_j)
                ious_1to1_class.append(best_val)
            else:
                ious_1to1_class.append(0.0)

    def summarize_ious(arr):
        a = np.array(arr, dtype=np.float64)
        return {
            "mIoU": round(float(np.mean(a) * 100.0), 2),
            "Acc@0.5": round(float(np.mean(a >= 0.5) * 100.0), 2),
            "Acc@0.7": round(float(np.mean(a >= 0.7) * 100.0), 2),
            "Acc@0.75": round(float(np.mean(a >= 0.75) * 100.0), 2),
            "count": int(len(a))
        }

    # AP calculations
    ap50_agnostic = evaluate_ap_for_threshold(samples, 0.50, class_aware=False)
    ap70_agnostic = evaluate_ap_for_threshold(samples, 0.70, class_aware=False)
    ap75_agnostic = evaluate_ap_for_threshold(samples, 0.75, class_aware=False)

    ap50_class = evaluate_ap_for_threshold(samples, 0.50, class_aware=True)
    ap70_class = evaluate_ap_for_threshold(samples, 0.70, class_aware=True)
    ap75_class = evaluate_ap_for_threshold(samples, 0.75, class_aware=True)

    # COCO mAP@50:95
    ap_50_95_ag = []
    ap_50_95_cl_weighted = []
    ap_50_95_cl_macro = []
    for th in np.arange(0.50, 0.96, 0.05):
        r_ag = evaluate_ap_for_threshold(samples, float(th), class_aware=False)
        r_cl = evaluate_ap_for_threshold(samples, float(th), class_aware=True)
        ap_50_95_ag.append(r_ag["ap_all_point"])
        ap_50_95_cl_weighted.append(r_cl["weighted_mAP"])
        ap_50_95_cl_macro.append(r_cl["macro_mAP"])

    results = {
        "dataset": "VRSBench Test Set (1,000 Images, 2,198 GT Objects, 3,539 Predicted Boxes)",
        "model": "Faster R-CNN (ResNet-50-FPN, Epoch 5 Checkpoint)",
        "grounding_and_localization_metrics": {
            "best_overlap_class_agnostic": summarize_ious(ious_best_agnostic),
            "best_overlap_class_matched": summarize_ious(ious_best_class),
            "one_to_one_class_agnostic": summarize_ious(ious_1to1_agnostic),
            "one_to_one_class_matched": summarize_ious(ious_1to1_class),
            "positive_detected_only_mIoU": summarize_ious(ious_detected_only)
        },
        "ap_metrics": {
            "micro_class_agnostic": {
                "mAP@50": round(ap50_agnostic["ap_all_point"], 2),
                "mAP@70": round(ap70_agnostic["ap_all_point"], 2),
                "mAP@75": round(ap75_agnostic["ap_all_point"], 2),
                "mAP@50:95": round(float(np.mean(ap_50_95_ag)), 2),
                "recall@50_1to1": round(ap50_agnostic["max_recall"], 2),
                "precision@50_1to1": round(ap50_agnostic["final_precision"], 2)
            },
            "class_matched_weighted": {
                "mAP@50": round(ap50_class["weighted_mAP"], 2),
                "mAP@70": round(ap70_class["weighted_mAP"], 2),
                "mAP@75": round(ap75_class["weighted_mAP"], 2),
                "mAP@50:95": round(float(np.mean(ap_50_95_cl_weighted)), 2)
            },
            "class_matched_macro_26_classes": {
                "mAP@50": round(ap50_class["macro_mAP"], 2),
                "mAP@70": round(ap70_class["macro_mAP"], 2),
                "mAP@75": round(ap75_class["macro_mAP"], 2),
                "mAP@50:95": round(float(np.mean(ap_50_95_cl_macro)), 2)
            },
            "per_class_ap50": ap50_class["per_class"]
        }
    }

    out_path = ROOT / "reports/VRSBENCH_DETECTOR_FULL_METRICS_1000.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results["grounding_and_localization_metrics"], indent=2))
    print(json.dumps({k: v for k, v in results["ap_metrics"].items() if k != "per_class_ap50"}, indent=2))

if __name__ == "__main__":
    main()

