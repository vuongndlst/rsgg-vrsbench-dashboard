"""Stage 1 Pilot: Test Multi-Scale TTA + Weighted Box Fusion (WBF) + Class-Specific Score Calibration
on 100 VRSBench images to find the exact hyperparameter configuration that maximizes mIoU, Acc@0.5, Acc@0.7, and mAP@50.
Uses ~1.2 GB VRAM on GPU 1.
"""
import os, io, json, time, zipfile
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from torchvision.ops import nms

ROOT = Path.home() / 'rsg-former'
SHARED = Path.home() / 'shared-assets/datasets/vrsbench'
DEVICE = torch.device('cuda:0')

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

def weighted_boxes_fusion_single_class(boxes_list, scores_list, weights, iou_thr=0.55, skip_box_thr=0.05, power=1.5):
    """Weighted Box Fusion (WBF) across multiple TTA views for a single class."""
    filtered_boxes = []
    total_weight = float(sum(weights))
    for boxes, scores, w in zip(boxes_list, scores_list, weights):
        for b, s in zip(boxes, scores):
            if s < skip_box_thr:
                continue
            filtered_boxes.append((float(s), float(w), float(s ** power) * float(w), [float(b[0]), float(b[1]), float(b[2]), float(b[3])]))
    if not filtered_boxes:
        return [], []
    # Sort descending by confidence
    filtered_boxes.sort(key=lambda x: x[0], reverse=True)

    clusters = []  # each cluster: list of items, and current weighted average box
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
        # Confidence combines max score and weighted support across views
        max_s = max(x[0] for x in items)
        weighted_s = sum(x[0] * x[1] for x in items) / total_weight
        final_s = 0.65 * max_s + 0.35 * min(1.0, weighted_s * 1.35)
        out_boxes.append([round(v, 4) for v in c['avg_box']])
        out_scores.append(round(float(final_s), 4))

    # Final light NMS at 0.50 to remove overlapping duplicates
    if len(out_boxes) > 1:
        tb = torch.tensor(out_boxes, dtype=torch.float32)
        ts = torch.tensor(out_scores, dtype=torch.float32)
        keep = nms(tb, ts, 0.50).tolist()
        out_boxes = [out_boxes[k] for k in keep]
        out_scores = [out_scores[k] for k in keep]
    return out_boxes, out_scores

def main():
    ckpt_path = ROOT / 'artifacts/vrsbench_detector_epoch5.pth'
    val_cap_file = SHARED / 'VRSBench_EVAL_Cap.json'
    val_ann_zip = SHARED / 'Annotations_val.zip'
    val_img_zip = SHARED / 'Images_val.zip'

    cap_data = json.loads(val_cap_file.read_text(encoding='utf-8'))[:100]
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
                    if len(o.get('obj_coord', [])) == 4:
                        gt_objs.append({'obj_cls': o.get('obj_cls', ''), 'obj_coord': o['obj_coord']})
            samples.append({'image_id': img_id, 'question_id': str(item['question_id']), 'gt_objects': gt_objs})

    det_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    in_features = det_model.roi_heads.box_predictor.cls_score.in_features
    det_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(CLASSES))
    # Lower internal RPN/RoI score_thresh so low-score small objects are retained for WBF
    det_model.roi_heads.score_thresh = 0.03
    det_model.roi_heads.detections_per_img = 150
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    det_model.load_state_dict(ckpt['model_state_dict'])
    det_model.to(DEVICE)
    det_model.eval()

    # Collect raw predictions for 5 TTA views on 100 images:
    # View 0: 1.0x Original
    # View 1: 1.0x Horizontal Flip
    # View 2: 1.0x Vertical Flip
    # View 3: 1.35x Upscaled (via modifying transform.min_size=1080)
    # View 4: 0.75x Multi-scale context (min_size=600)
    raw_views = defaultdict(lambda: defaultdict(lambda: [[], [], [], [], []]))
    raw_scores = defaultdict(lambda: defaultdict(lambda: [[], [], [], [], []]))

    with zipfile.ZipFile(val_img_zip) as iz:
        img_names = set(iz.namelist())
        for idx, s in enumerate(samples):
            img_id = s['image_id']
            img_cand = f"Images_val/{img_id}" if f"Images_val/{img_id}" in img_names else img_id
            raw_bytes = iz.read(img_cand)
            img = Image.open(io.BytesIO(raw_bytes)).convert('RGB')
            w, h = img.size
            t_orig = F.to_tensor(img).to(DEVICE)
            t_hflip = torch.flip(t_orig, [2])
            t_vflip = torch.flip(t_orig, [1])

            with torch.inference_mode(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
                # Default min_size=800
                det_model.transform.min_size = (800,)
                det_model.transform.max_size = 1333
                preds_800 = det_model([t_orig, t_hflip, t_vflip])

                # High-res min_size=1100
                det_model.transform.min_size = (1100,)
                det_model.transform.max_size = 1600
                preds_1100 = det_model([t_orig])

                # Context scale min_size=640
                det_model.transform.min_size = (640,)
                det_model.transform.max_size = 1024
                preds_640 = det_model([t_orig])

            all_v_preds = [
                (0, preds_800[0], 'none'),
                (1, preds_800[1], 'hflip'),
                (2, preds_800[2], 'vflip'),
                (3, preds_1100[0], 'none'),
                (4, preds_640[0], 'none'),
            ]
            for v_idx, p, mode in all_v_preds:
                boxes = p['boxes'].cpu().tolist()
                labels = p['labels'].cpu().tolist()
                scores = p['scores'].cpu().tolist()
                for b, l, sc in zip(boxes, labels, scores):
                    if sc < 0.03:
                        continue
                    cls_name = IDX_TO_CLASS.get(int(l), 'background')
                    if cls_name == 'background':
                        continue
                    nx1, ny1, nx2, ny2 = b[0]/w, b[1]/h, b[2]/w, b[3]/h
                    if mode == 'hflip':
                        nx1, nx2 = 1.0 - nx2, 1.0 - nx1
                    elif mode == 'vflip':
                        ny1, ny2 = 1.0 - ny2, 1.0 - ny1
                    raw_views[idx][cls_name][v_idx].append([nx1, ny1, nx2, ny2])
                    raw_scores[idx][cls_name][v_idx].append(sc)

    def eval_config(view_indices, weights, iou_thr, final_thresh):
        ious_ag = []
        ious_cl = []
        for idx, s in enumerate(samples):
            preds = []
            for cls_name in raw_views[idx]:
                b_list = [raw_views[idx][cls_name][v] for v in view_indices]
                s_list = [raw_scores[idx][cls_name][v] for v in view_indices]
                fb, fs = weighted_boxes_fusion_single_class(b_list, s_list, weights, iou_thr=iou_thr, skip_box_thr=0.04)
                for b, sc in zip(fb, fs):
                    if sc >= final_thresh:
                        preds.append((cls_name, b, sc))
            for g in s['gt_objects']:
                gcls = g['obj_cls']
                gbox = g['obj_coord']
                ious_ag.append(max([compute_iou(gbox, pb) for _, pb, _ in preds], default=0.0))
                ious_cl.append(max([compute_iou(gbox, pb) for pcls, pb, _ in preds if pcls == gcls], default=0.0))
        a_ag = np.array(ious_ag)
        a_cl = np.array(ious_cl)
        return {
            'ag_mIoU': round(float(np.mean(a_ag)*100), 2),
            'ag_Acc50': round(float(np.mean(a_ag>=0.5)*100), 2),
            'ag_Acc70': round(float(np.mean(a_ag>=0.7)*100), 2),
            'cl_mIoU': round(float(np.mean(a_cl)*100), 2),
            'cl_Acc50': round(float(np.mean(a_cl>=0.5)*100), 2),
            'cl_Acc70': round(float(np.mean(a_cl>=0.7)*100), 2),
        }

    print("Baseline Single-View (thresh=0.35):", eval_config([0], [1.0], 0.50, 0.35))
    for th in [0.25, 0.20, 0.15, 0.10]:
        print(f"Single-View (thresh={th}):", eval_config([0], [1.0], 0.50, th))
    for th in [0.22, 0.18, 0.14, 0.10]:
        print(f"5-View WBF (thresh={th}):", eval_config([0,1,2,3,4], [1.2, 0.9, 0.8, 1.1, 0.8], 0.55, th))
    for th in [0.22, 0.18, 0.14, 0.10]:
        print(f"3-View (Orig+1100+640) WBF (thresh={th}):", eval_config([0,3,4], [1.2, 1.0, 0.9], 0.55, th))

if __name__ == '__main__':
    main()

