"""Full 9,350-Image Official VRSBench Test Split Evaluation for Object Detection & Visual Grounding:
Evaluates across ALL 9,350 test images in Annotations_val.zip / Images_val.zip (all 26 classes).
Supports:
  1. Single-View High-Speed Evaluation (--mode single): ~12 mins on GPU 0
  2. Multi-View TTA + Weighted Box Fusion (WBF) Evaluation (--mode fast3view): ~45 mins
Computes exact mIoU, mAP@50 (across all 26 classes), Acc@0.5, Acc@0.7, Acc@0.75.
"""
import os, sys, io, json, time, zipfile, argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from torchvision.ops import nms

ROOT = Path(__file__).resolve().parent.parent if not (Path.home() / 'rsg-former').exists() else (Path.home() / 'rsg-former')
SHARED = ROOT / 'data/vrsbench' if not (Path.home() / 'shared-assets/datasets/vrsbench').exists() else (Path.home() / 'shared-assets/datasets/vrsbench')

CLASSES = [
    'background', 'vehicle', 'airplane', 'ship', 'harbor', 'bridge',
    'tennis-court', 'storage-tank', 'baseball-diamond', 'swimming-pool',
    'ground-track-field', 'basketball-court', 'roundabout', 'windmill',
    'overpass', 'soccer-ball-field', 'expressway-service-area', 'airport',
    'trainstation', 'expressway-toll-station', 'chimney', 'stadium', 'dam',
    'golffield', 'helicopter', 'container-crane', 'helipad'
]
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    a2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    u = a1 + a2 - inter
    return inter / u if u > 0 else 0.0

def normalize_cls(cls_name):
    c = (cls_name or '').strip().lower()
    for p in ['small ', 'large ', 'small-', 'large-']:
        if c.startswith(p):
            c = c[len(p):]
    return c

def weighted_boxes_fusion_single_class(boxes_list, scores_list, weights, iou_thr=0.55, skip_box_thr=0.04, power=1.6):
    filtered_boxes = []
    total_weight = float(sum(weights))
    for boxes, scores, w in zip(boxes_list, scores_list, weights):
        for b, s in zip(boxes, scores):
            if s < skip_box_thr:
                continue
            filtered_boxes.append((float(s), float(w), float(s ** power) * float(w), [float(b[0]), float(b[1]), float(b[2]), float(b[3])]))
    if not filtered_boxes:
        return [], []
    filtered_boxes.sort(key=lambda x: x[0], reverse=True)

    clusters = []
    for item in filtered_boxes:
        s, w, sw, b = item
        best_iou = -1.0
        best_c = -1
        for ci, c in enumerate(clusters):
            iou = compute_iou(b, c['avg_box'])
            if iou > best_iou:
                best_iou = iou
                best_c = ci
        if best_iou >= iou_thr and best_c >= 0:
            c = clusters[best_c]
            c['items'].append(item)
            w_sum = sum(x[2] for x in c['items'])
            c['avg_box'] = [
                sum(x[2] * x[3][0] for x in c['items']) / w_sum,
                sum(x[2] * x[3][1] for x in c['items']) / w_sum,
                sum(x[2] * x[3][2] for x in c['items']) / w_sum,
                sum(x[2] * x[3][3] for x in c['items']) / w_sum,
            ]
        else:
            clusters.append({'items': [item], 'avg_box': list(b)})

    out_boxes = []
    out_scores = []
    for c in clusters:
        items = c['items']
        max_s = max(x[0] for x in items)
        weighted_s = sum(x[0] * x[1] for x in items) / total_weight
        final_s = min(0.9999, 0.55 * max_s + 0.45 * min(1.0, weighted_s * 1.40))
        out_boxes.append([round(max(0.0, min(1.0, v)), 4) for v in c['avg_box']])
        out_scores.append(round(float(final_s), 4))

    if len(out_boxes) > 1:
        tb = torch.tensor(out_boxes, dtype=torch.float32)
        ts = torch.tensor(out_scores, dtype=torch.float32)
        keep = nms(tb, ts, 0.50).tolist()
        out_boxes = [out_boxes[k] for k in keep]
        out_scores = [out_scores[k] for k in keep]
    return out_boxes, out_scores

def compute_ap_all_point(recalls, precisions):
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))

def evaluate_ap_full(samples, iou_thresh=0.5, class_aware=True):
    if not class_aware:
        gt_by_img = {}
        all_preds = []
        total_gt = 0
        for s in samples:
            qid = str(s['question_id'])
            gts = [g['obj_coord'] for g in s.get('gt_objects', []) if len(g.get('obj_coord', [])) == 4]
            gt_by_img[qid] = {'boxes': gts, 'matched': [False] * len(gts)}
            total_gt += len(gts)
            for p in s.get('preds', []):
                all_preds.append((float(p['score']), qid, p['obj_coord']))
        all_preds.sort(key=lambda x: x[0], reverse=True)
        tp = np.zeros(len(all_preds))
        fp = np.zeros(len(all_preds))
        for i, (sc, qid, pbox) in enumerate(all_preds):
            ginfo = gt_by_img[qid]
            best_iou = 0.0
            best_j = -1
            for j, gbox in enumerate(ginfo['boxes']):
                iou = compute_iou(pbox, gbox)
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            if best_iou >= iou_thresh and best_j >= 0 and not ginfo['matched'][best_j]:
                tp[i] = 1.0
                ginfo['matched'][best_j] = True
            else:
                fp[i] = 1.0
        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        recalls = cum_tp / max(1, total_gt)
        precisions = cum_tp / np.maximum(1e-9, cum_tp + cum_fp)
        return {
            'ap_all_point': round(compute_ap_all_point(recalls, precisions) * 100.0, 2),
            'max_recall': round(float(recalls[-1] * 100.0), 2) if len(recalls) else 0.0,
        }

    classes = sorted(set(
        normalize_cls(g.get('obj_cls', ''))
        for s in samples for g in s.get('gt_objects', [])
        if len(g.get('obj_coord', [])) == 4 and normalize_cls(g.get('obj_cls', ''))
    ))
    per_class = {}
    for cls in classes:
        gt_by_img = {}
        all_preds = []
        total_gt = 0
        for s in samples:
            qid = str(s['question_id'])
            gts = [
                g['obj_coord'] for g in s.get('gt_objects', [])
                if len(g.get('obj_coord', [])) == 4 and normalize_cls(g.get('obj_cls', '')) == cls
            ]
            gt_by_img[qid] = {'boxes': gts, 'matched': [False] * len(gts)}
            total_gt += len(gts)
            for p in s.get('preds', []):
                if normalize_cls(p.get('obj_cls', '')) == cls:
                    all_preds.append((float(p['score']), qid, p['obj_coord']))
        if total_gt == 0:
            continue
        all_preds.sort(key=lambda x: x[0], reverse=True)
        tp = np.zeros(len(all_preds))
        fp = np.zeros(len(all_preds))
        for i, (sc, qid, pbox) in enumerate(all_preds):
            ginfo = gt_by_img[qid]
            best_iou = 0.0
            best_j = -1
            for j, gbox in enumerate(ginfo['boxes']):
                iou = compute_iou(pbox, gbox)
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            if best_iou >= iou_thresh and best_j >= 0 and not ginfo['matched'][best_j]:
                tp[i] = 1.0
                ginfo['matched'][best_j] = True
            else:
                fp[i] = 1.0
        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        recalls = cum_tp / max(1, total_gt)
        precisions = cum_tp / np.maximum(1e-9, cum_tp + cum_fp)
        per_class[cls] = {
            'ap_all_point': round(compute_ap_all_point(recalls, precisions) * 100.0, 2),
            'recall': round(float(recalls[-1] * 100.0), 2) if len(recalls) else 0.0,
            'gt_count': total_gt,
            'tp': int(np.sum(tp))
        }
    macro_ap = float(np.mean([v['ap_all_point'] for v in per_class.values()])) if per_class else 0.0
    weighted_ap = float(
        np.sum([v['ap_all_point'] * v['gt_count'] for v in per_class.values()])
        / max(1, sum(v['gt_count'] for v in per_class.values()))
    ) if per_class else 0.0
    return {
        'macro_mAP': round(macro_ap, 2),
        'weighted_mAP': round(weighted_ap, 2),
        'num_active_classes': len(per_class),
        'per_class': per_class
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--ckpt', default='artifacts/vrsbench_detector_v2_best.pth')
    parser.add_argument('--arch', default='v2', choices=['v1', 'v2'])
    parser.add_argument('--mode', default='single', choices=['single', 'fast3view'])
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    device = torch.device('cuda:0')
    t0 = time.time()

    val_cap_file = SHARED / 'VRSBench_EVAL_Cap.json'
    val_ann_zip = SHARED / 'Annotations_val.zip'
    val_img_zip = SHARED / 'Images_val.zip'

    cap_data = json.loads(val_cap_file.read_text(encoding='utf-8'))
    samples = []
    with zipfile.ZipFile(val_ann_zip) as az:
        ann_names = set(az.namelist())
        for item in cap_data:
            img_id = item['image_id']
            stem = Path(img_id).stem
            ann_cand = f"Annotations_val/{stem}.json" if f"Annotations_val/{stem}.json" in ann_names else f"{stem}.json"
            gt_objs = []
            if ann_cand in ann_names:
                adata = json.loads(az.read(ann_cand).decode('utf-8'))
                for o in adata.get('objects', []):
                    coord = o.get('obj_coord')
                    if not coord or len(coord) != 4:
                        corner = o.get('obj_corner')
                        if corner and len(corner) == 8:
                            xs = [float(corner[i]) for i in range(0, 8, 2)]
                            ys = [float(corner[i]) for i in range(1, 8, 2)]
                            coord = [min(xs), min(ys), max(xs), max(ys)]
                    if coord and len(coord) == 4:
                        gt_objs.append({'obj_cls': o.get('obj_cls', ''), 'obj_coord': [float(x) for x in coord]})
            samples.append({'image_id': img_id, 'question_id': str(item['question_id']), 'gt_objects': gt_objs})

    print(f"Indexed ALL {len(samples)} official VRSBench test images ({sum(len(s['gt_objects']) for s in samples)} GT boxes).", flush=True)

    if args.arch == 'v2':
        det_model = fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None)
    else:
        det_model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    in_features = det_model.roi_heads.box_predictor.cls_score.in_features
    det_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(CLASSES))
    det_model.roi_heads.score_thresh = 0.04
    det_model.roi_heads.detections_per_img = 150
    ckpt = torch.load(ROOT / args.ckpt, map_location=device)
    det_model.load_state_dict(ckpt['model_state_dict'])
    det_model.to(device)
    det_model.eval()

    print(f"Running evaluation with arch={args.arch}, mode={args.mode} on GPU {args.gpu}...", flush=True)
    weights = [1.25, 0.95, 1.10]  # Orig (800), H-Flip (800), High-Res (1050)
    with zipfile.ZipFile(val_img_zip) as iz:
        img_names = set(iz.namelist())
        for idx, s in enumerate(samples):
            img_id = s['image_id']
            img_cand = f"Images_val/{img_id}" if f"Images_val/{img_id}" in img_names else img_id
            raw_bytes = iz.read(img_cand)
            img = Image.open(io.BytesIO(raw_bytes)).convert('RGB')
            w, h = img.size
            t_orig = F.to_tensor(img).to(device)

            if args.mode == 'single':
                with torch.inference_mode(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    det_model.transform.min_size = (800,)
                    det_model.transform.max_size = 1333
                    p = det_model([t_orig])[0]

                objs = []
                for b, l, sc in zip(p['boxes'].cpu().tolist(), p['labels'].cpu().tolist(), p['scores'].cpu().tolist()):
                    if sc < 0.10:
                        continue
                    cname = IDX_TO_CLASS.get(int(l), 'background')
                    if cname == 'background':
                        continue
                    objs.append({
                        'obj_cls': cname,
                        'obj_coord': [round(b[0]/w, 4), round(b[1]/h, 4), round(b[2]/w, 4), round(b[3]/h, 4)],
                        'score': round(sc, 4)
                    })
                objs.sort(key=lambda x: x['score'], reverse=True)
                s['preds'] = objs

            elif args.mode == 'fast3view':
                t_hflip = torch.flip(t_orig, [2])
                with torch.inference_mode(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    det_model.transform.min_size = (800,)
                    det_model.transform.max_size = 1333
                    p800 = det_model([t_orig, t_hflip])

                    det_model.transform.min_size = (1050,)
                    det_model.transform.max_size = 1500
                    p1050 = det_model([t_orig])

                views = [(0, p800[0], 'none'), (1, p800[1], 'hflip'), (2, p1050[0], 'none')]
                cls_b = defaultdict(lambda: [[], [], []])
                cls_s = defaultdict(lambda: [[], [], []])
                for v_idx, p, mode in views:
                    for b, l, sc in zip(p['boxes'].cpu().tolist(), p['labels'].cpu().tolist(), p['scores'].cpu().tolist()):
                        if sc < 0.04:
                            continue
                        cname = IDX_TO_CLASS.get(int(l), 'background')
                        if cname == 'background':
                            continue
                        nx1, ny1, nx2, ny2 = b[0]/w, b[1]/h, b[2]/w, b[3]/h
                        if mode == 'hflip':
                            nx1, nx2 = 1.0 - nx2, 1.0 - nx1
                        cls_b[cname][v_idx].append([nx1, ny1, nx2, ny2])
                        cls_s[cname][v_idx].append(sc)

                objs = []
                for cname in cls_b:
                    fb, fs = weighted_boxes_fusion_single_class(cls_b[cname], cls_s[cname], weights, iou_thr=0.55, skip_box_thr=0.04)
                    for b, sc in zip(fb, fs):
                        if sc >= 0.10:
                            objs.append({'obj_cls': cname, 'obj_coord': b, 'score': sc})
                objs.sort(key=lambda x: x['score'], reverse=True)
                s['preds'] = objs

            if (idx + 1) % 500 == 0 or (idx + 1) == len(samples):
                rate = (idx + 1) / max(0.1, time.time() - t0)
                rem_sec = (len(samples) - (idx + 1)) / max(0.01, rate)
                print(f"  Full-9350 Detector Eval [{args.mode}]: [{idx+1}/{len(samples)}] ({time.time()-t0:.1f}s, {rate:.1f} img/s, ETA: {rem_sec/60:.1f}m)", flush=True)

    ious_ag = []
    ious_cl = []
    for s in samples:
        preds = [p for p in s['preds'] if p['score'] >= 0.10]
        for g in s['gt_objects']:
            gcls = normalize_cls(g['obj_cls'])
            gbox = g['obj_coord']
            ious_ag.append(max([compute_iou(gbox, p['obj_coord']) for p in preds], default=0.0))
            ious_cl.append(max([compute_iou(gbox, p['obj_coord']) for p in preds if normalize_cls(p['obj_cls']) == gcls], default=0.0))

    a_ag = np.array(ious_ag)
    a_cl = np.array(ious_cl)
    ap50_cl = evaluate_ap_full(samples, 0.50, class_aware=True)
    ap50_ag = evaluate_ap_full(samples, 0.50, class_aware=False)

    res = {
        'task': f'Full 9,350-Image VRSBench Detection & Grounding Evaluation ({args.arch}, {args.ckpt}, mode={args.mode})',
        'mode': args.mode,
        'checkpoint': args.ckpt,
        'total_images': len(samples),
        'total_gt_boxes': len(ious_ag),
        'localization_agnostic': {
            'mIoU': round(float(np.mean(a_ag)*100), 2),
            'Acc@0.5': round(float(np.mean(a_ag>=0.5)*100), 2),
            'Acc@0.7': round(float(np.mean(a_ag>=0.7)*100), 2),
            'Acc@0.75': round(float(np.mean(a_ag>=0.75)*100), 2),
        },
        'class_matched': {
            'mIoU': round(float(np.mean(a_cl)*100), 2),
            'Acc@0.5': round(float(np.mean(a_cl>=0.5)*100), 2),
            'Acc@0.7': round(float(np.mean(a_cl>=0.7)*100), 2),
            'Acc@0.75': round(float(np.mean(a_cl>=0.75)*100), 2),
        },
        'ap_metrics': {
            'macro_mAP@50_all_26_classes': ap50_cl['macro_mAP'],
            'weighted_mAP@50': ap50_cl['weighted_mAP'],
            'micro_mAP@50': ap50_ag['ap_all_point'],
            'num_active_classes': ap50_cl['num_active_classes'],
            'per_class_ap50': ap50_cl['per_class']
        }
    }
    mode_tag = 'SINGLE' if args.mode == 'single' else 'WBF'
    out_file = ROOT / f"reports/VRSBENCH_DETECTOR_FULL9350_{args.arch.upper()}_{mode_tag}_REPORT.json"
    out_file.write_text(json.dumps(res, indent=2), encoding='utf-8')

    # Save boxes for downstream scene graph and captioning
    boxes_file = ROOT / f"reports/VRSBENCH_DETECTOR_{args.arch.upper()}_{mode_tag}_BOXES_9350.json"
    boxes_dict = {
        'total_images': len(samples),
        'detections': {
            s['question_id']: {
                'image_id': s['image_id'],
                'detected_objects': s['preds']
            }
            for s in samples
        }
    }
    boxes_file.write_text(json.dumps(boxes_dict), encoding='utf-8')
    print(f"Saved predictions to {boxes_file.name}!", flush=True)

    print("============================================================")
    print(json.dumps({k: v for k, v in res.items() if k != 'ap_metrics'}, indent=2))
    print(json.dumps({k: v for k, v in res['ap_metrics'].items() if k != 'per_class_ap50'}, indent=2))
    print("============================================================")

if __name__ == '__main__':
    main()
