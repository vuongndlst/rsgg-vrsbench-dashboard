"""End-to-End Evaluation on the Exact Same 1,000 VRSBench Test Images:
Trained Faster R-CNN Detector -> Predicted Scene Graph -> Qwen2.5-14B LoRA Captioning -> Compare vs Oracle GT SG.
Runs on GPU 1 or GPU 0.
"""
import os, sys, time, json, zipfile, io, argparse
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from pathlib import Path
from collections import Counter
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from torchvision.ops import box_iou, nms
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--detector-ckpt', default='artifacts/vrsbench_detector_epoch5.pth', help='Detector checkpoint path')
    parser.add_argument('--adapter', default='qwen_vrsbench_g2t_full_step900', help='Qwen LoRA adapter in artifacts/')
    parser.add_argument('--num-samples', type=int, default=1000, help='Exact same 1,000 test samples')
    parser.add_argument('--score-thresh', type=float, default=0.35, help='Detection confidence threshold')
    parser.add_argument('--nms-thresh', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--gpu', default='1', help='CUDA_VISIBLE_DEVICES')
    parser.add_argument('--out-report', default='reports/VRSBENCH_DETECTOR_SG_1000_EVAL_REPORT.json', help='Output JSON report')
    return parser.parse_args()

CLASSES = [
    'background', 'vehicle', 'airplane', 'ship', 'harbor', 'bridge',
    'tennis-court', 'storage-tank', 'baseball-diamond', 'swimming-pool',
    'ground-track-field', 'basketball-court', 'roundabout', 'windmill',
    'overpass', 'soccer-ball-field', 'expressway-service-area', 'airport',
    'trainstation', 'expressway-toll-station', 'chimney', 'stadium', 'dam',
    'golffield', 'helicopter', 'container-crane', 'helipad'
]
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}

def get_spatial_phrase(cx, cy):
    """Map normalized center (cx, cy) in [0, 1] to natural remote sensing spatial region."""
    if cy < 0.30:
        v = "top"
    elif cy > 0.70:
        v = "bottom"
    else:
        v = "middle"

    if cx < 0.30:
        h = "left"
    elif cx > 0.70:
        h = "right"
    else:
        h = "center"

    if v == "top" and h == "left":
        return "near the top-left corner of the image"
    if v == "top" and h == "right":
        return "near the top-right corner of the image"
    if v == "bottom" and h == "left":
        return "near the bottom-left corner of the image"
    if v == "bottom" and h == "right":
        return "near the bottom-right corner of the image"
    if v == "top" and h == "center":
        return "at the top edge of the image"
    if v == "bottom" and h == "center":
        return "at the bottom side of the image"
    if v == "middle" and h == "left":
        return "on the left side of the image"
    if v == "middle" and h == "right":
        return "on the right side of the image"
    return "in the center of the image"

def build_scene_graph_from_detections(det_objs):
    """Convert detected bounding boxes into a structured VRSBench Scene Graph string."""
    if not det_objs:
        return "aerial scene overview"

    # Sort detections by confidence descending, cap at 12 salient objects to avoid noisy clutter
    sorted_objs = sorted(det_objs, key=lambda x: x['score'], reverse=True)[:12]
    obj_summary = []
    for o in sorted_objs:
        cls_name = o['obj_cls']
        x1, y1, x2, y2 = o['obj_coord']
        area = max(0.0, (x2 - x1) * (y2 - y1))
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

        if area >= 0.06:
            size = "large"
        elif area <= 0.012:
            size = "small"
        else:
            size = ""

        region = get_spatial_phrase(cx, cy)
        size_prefix = f"{size} " if size else ""
        ref = f"The {size_prefix}{cls_name} is located {region}"
        item_desc = f"{size_prefix}{cls_name} ({ref})"
        obj_summary.append(item_desc)

    return "; ".join(obj_summary)

def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    DEVICE = torch.device('cuda:0')
    torch.cuda.set_device(DEVICE)

    ROOT = Path.home() / 'rsg-former'
    SHARED = Path.home() / 'shared-assets/datasets/vrsbench'
    sys.path.insert(0, str(ROOT / '.tools/deps'))
    sys.path.insert(0, str(ROOT / 'src'))

    java_bin = os.path.expanduser('~/rsg-former/.tools/java/temurin8-52dcd578baca/jdk8u504-b01-jre/bin')
    if os.path.exists(java_bin):
        os.environ['PATH'] = java_bin + ':' + os.environ['PATH']

    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.cider.cider import Cider

    print(f"[{time.strftime('%X')}] Starting End-to-End Detector -> SG -> Qwen2.5-14B Evaluation on {args.num_samples} images...", flush=True)

    # 1. Load 1,000 Test Samples
    eval_path = SHARED / 'VRSBench_EVAL_Cap.json'
    val_ann_zip = SHARED / 'Annotations_val.zip'
    val_img_zip = SHARED / 'Images_val.zip'
    with open(eval_path) as f:
        eval_data = json.load(f)

    target_samples = eval_data[:args.num_samples]
    test_samples = []
    with zipfile.ZipFile(val_ann_zip) as az:
        for item in target_samples:
            img_id = item.get('image_id', '')
            base_name = img_id.replace('.png', '').replace('.jpg', '')
            cand = f"Annotations_val/{base_name}.json"
            gt_objs = []
            oracle_summary = []
            gt_text = item.get('ground_truth', '')
            try:
                data = json.loads(az.read(cand).decode('utf-8'))
                if not gt_text:
                    gt_text = data.get('caption', '')
                for o in data.get('objects', []):
                    cls_name = o.get('obj_cls', 'object')
                    size = o.get('obj_size', '')
                    ref = o.get('referring_sentence', '')
                    coord = o.get('obj_coord', [])
                    if not coord and len(o.get('obj_corner', [])) == 8:
                        c = o['obj_corner']
                        coord = [min(c[0::2]), min(c[1::2]), max(c[0::2]), max(c[1::2])]
                    gt_objs.append({
                        'obj_cls': cls_name,
                        'obj_coord': coord,
                        'obj_size': size,
                        'referring_sentence': ref
                    })
                    item_desc = f"{size + ' ' if size else ''}{cls_name}"
                    if ref:
                        item_desc += f" ({ref.strip('.')})"
                    oracle_summary.append(item_desc)
            except Exception:
                pass

            test_samples.append({
                'question_id': str(item.get('question_id')),
                'image_id': img_id,
                'ground_truth': gt_text,
                'oracle_graph': "; ".join(oracle_summary) if oracle_summary else "aerial scene overview",
                'gt_objects': gt_objs
            })

    # 2. Run Trained Object Detector on all 1,000 test images (or load from saved cache)
    ckpt_path = ROOT / args.detector_ckpt
    boxes_cache_file = ROOT / "reports/VRSBENCH_DETECTOR_BOXES_1000.json"
    if boxes_cache_file.exists():
        print(f"Loading cached 1,000 detector boxes & scene graphs from {boxes_cache_file.name}...", flush=True)
        bcache = json.loads(boxes_cache_file.read_text(encoding='utf-8'))
        dm = bcache.get('detection_metrics', {})
        total_gt_boxes = dm.get('total_gt_boxes', 0)
        total_pred_boxes = dm.get('total_pred_boxes', 0)
        det_recall = dm.get('recall_at_50', 0.0)
        det_precision = dm.get('precision_at_50', 0.0)
        det_mean_iou = dm.get('mean_max_iou', 0.0)
        det_map = bcache.get('detections', {})
        for s in test_samples:
            d = det_map.get(s['question_id'], {})
            s['detected_objects'] = d.get('detected_objects', [])
            s['predicted_graph'] = d.get('predicted_graph', 'aerial scene overview')
        print(f"Loaded 1,000 detections! Recall@0.5={det_recall}%, Precision@0.5={det_precision}%, MeanMaxIoU={det_mean_iou}%", flush=True)
    else:
        if not ckpt_path.exists():
            for ep in range(5, 0, -1):
                alt = ROOT / f"artifacts/vrsbench_detector_epoch{ep}.pth"
                if alt.exists():
                    ckpt_path = alt
                    break
        assert ckpt_path.exists(), f"No detector checkpoint found at {ckpt_path}"
        print(f"Loading Trained Faster R-CNN Detector from {ckpt_path}...", flush=True)

        det_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
        in_features = det_model.roi_heads.box_predictor.cls_score.in_features
        det_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(CLASSES))
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        det_model.load_state_dict(ckpt['model_state_dict'])
        det_model.to(DEVICE)
        det_model.eval()

        total_gt_boxes = 0
        total_pred_boxes = 0
        matched_gt_boxes = 0
        iou_sum = 0.0

        t_det = time.time()
        with zipfile.ZipFile(val_img_zip) as iz:
            img_namelist = set(iz.namelist())
            for idx, s in enumerate(test_samples):
                img_id = s['image_id']
                img_cand = f"Images_val/{img_id}" if f"Images_val/{img_id}" in img_namelist else img_id
                det_objs = []
                if img_cand in img_namelist:
                    raw_bytes = iz.read(img_cand)
                    img = Image.open(io.BytesIO(raw_bytes)).convert('RGB')
                    w, h = img.size
                    img_t = F.to_tensor(img).to(DEVICE)
                    with torch.inference_mode():
                        pred = det_model([img_t])[0]

                    boxes = pred['boxes']
                    labels = pred['labels']
                    scores = pred['scores']

                    keep_score = scores >= args.score_thresh
                    boxes = boxes[keep_score]
                    labels = labels[keep_score]
                    scores = scores[keep_score]

                    if len(boxes) > 0:
                        keep_nms = nms(boxes, scores, args.nms_thresh)
                        boxes = boxes[keep_nms]
                        labels = labels[keep_nms]
                        scores = scores[keep_nms]

                    for b, l, sc in zip(boxes.cpu().tolist(), labels.cpu().tolist(), scores.cpu().tolist()):
                        cls_name = IDX_TO_CLASS.get(int(l), 'object')
                        if cls_name == 'background':
                            continue
                        norm_box = [round(b[0]/w, 4), round(b[1]/h, 4), round(b[2]/w, 4), round(b[3]/h, 4)]
                        det_objs.append({
                            'obj_cls': cls_name,
                            'obj_coord': norm_box,
                            'score': round(float(sc), 4)
                        })

                    gt_coords = [g['obj_coord'] for g in s['gt_objects'] if len(g.get('obj_coord', [])) == 4]
                    total_gt_boxes += len(gt_coords)
                    total_pred_boxes += len(det_objs)
                    if gt_coords and det_objs:
                        gt_t = torch.tensor(gt_coords, dtype=torch.float32)
                        pr_t = torch.tensor([d['obj_coord'] for d in det_objs], dtype=torch.float32)
                        ious = box_iou(gt_t, pr_t)
                        max_ious, _ = ious.max(dim=1)
                        hits = (max_ious >= 0.5)
                        matched_gt_boxes += int(hits.sum().item())
                        iou_sum += float(max_ious.sum().item())

                s['detected_objects'] = det_objs
                s['predicted_graph'] = build_scene_graph_from_detections(det_objs)

                if (idx + 1) % 200 == 0 or (idx + 1) == len(test_samples):
                    print(f"  Detector Inference: [{idx+1}/{len(test_samples)}] ({time.time()-t_det:.1f}s)", flush=True)

        del det_model
        torch.cuda.empty_cache()

        det_recall = round(matched_gt_boxes / max(1, total_gt_boxes) * 100, 2)
        det_precision = round(matched_gt_boxes / max(1, total_pred_boxes) * 100, 2)
        det_mean_iou = round(iou_sum / max(1, total_gt_boxes) * 100, 2)
        print(f"Detector Summary on {len(test_samples)} images: Recall@0.5={det_recall}%, Precision@0.5={det_precision}%, MeanMaxIoU={det_mean_iou}%", flush=True)

        with open(boxes_cache_file, 'w', encoding='utf-8') as bf:
            json.dump({
                'detection_metrics': {
                    'total_gt_boxes': total_gt_boxes,
                    'total_pred_boxes': total_pred_boxes,
                    'recall_at_50': det_recall,
                    'precision_at_50': det_precision,
                    'mean_max_iou': det_mean_iou
                },
                'detections': {
                    s['question_id']: {
                        'detected_objects': s['detected_objects'],
                        'predicted_graph': s['predicted_graph']
                    } for s in test_samples
                }
            }, bf, indent=2)
        print(f"Saved 1,000 detector box & scene graph predictions to {boxes_cache_file}", flush=True)

    # 3. Generate Captions from Predicted Scene Graphs using Qwen2.5-14B LoRA
    cache_file = ROOT / "reports/VRSBENCH_QWEN_G2T_DETECTOR_CAPTIONS_CACHE.json"
    captions = {}
    if cache_file.exists():
        try:
            captions = json.loads(cache_file.read_text(encoding='utf-8'))
            print(f"Loaded {len(captions)} cached Detector-SG captions.", flush=True)
        except Exception:
            captions = {}

    remaining = [s for s in test_samples if s['question_id'] not in captions]
    print(f"Generating Qwen2.5-14B captions from Predicted SG for {len(remaining)} / {len(test_samples)} samples...", flush=True)

    t_gen = time.time()
    if len(remaining) > 0:
        model_path = '/home/stackops/etechs/msvd-tkg-captioning/models/qwen2.5-14b'
        adapter_path = ROOT / 'artifacts' / args.adapter
        tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        if tok.pad_token_id is None:
            tok.pad_token_id = tok.eos_token_id

        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            local_files_only=True
        ).to(DEVICE)
        peft_model = PeftModel.from_pretrained(base_model, str(adapter_path)).to(DEVICE)
        model = peft_model.merge_and_unload()
        model.eval()

        SYS_MSG = (
            "You are an expert satellite remote sensing analyst. "
            "Given the detected geospatial entities and spatial layout from the aerial scene, "
            "generate a comprehensive, precise, and professional description of the image."
        )

        for idx, s in enumerate(remaining):
            user_prompt = f"[Remote Sensing Scene Graph]\nEntities & Layout: {s['predicted_graph']}\nTask: Describe the aerial scene in detail."
            msg = [{"role": "system", "content": SYS_MSG}, {"role": "user", "content": user_prompt}]
            prompt_text = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            input_ids = tok.encode(prompt_text, return_tensors='pt').to(DEVICE)
            attn_mask = torch.ones_like(input_ids, device=DEVICE)
            with torch.inference_mode():
                out = model.generate(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    max_new_tokens=75,
                    do_sample=False,
                    repetition_penalty=1.1,
                    pad_token_id=tok.pad_token_id,
                    use_cache=True
                )
            gen = tok.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
            captions[s['question_id']] = gen

            done_now = idx + 1
            total_done = len(test_samples) - len(remaining) + done_now
            if done_now % 25 == 0 or done_now == len(remaining):
                el = time.time() - t_gen
                rate = el / done_now
                rem = (len(remaining) - done_now) * rate / 60.0
                print(f"  Detector-SG Caption Progress: [{total_done}/{len(test_samples)} - {total_done/len(test_samples)*100:.1f}%] | Rate: {rate:.2f}s/sample | ETA: {rem:.1f}m", flush=True)
                with open(cache_file, 'w', encoding='utf-8') as cf:
                    json.dump(captions, cf, indent=2)

    # 4. Compute Official pycocoevalcap Metrics
    gts_raw = {s['question_id']: [{'caption': s['ground_truth']}] for s in test_samples}
    eval_caps = {s['question_id']: [{'caption': captions[s['question_id']]}] for s in test_samples}
    tokenizer = PTBTokenizer()
    gts_tok = tokenizer.tokenize(gts_raw)
    res_tok = tokenizer.tokenize(eval_caps)

    b_score, b_ind = Bleu(4).compute_score(gts_tok, res_tok)
    m_scorer = Meteor()
    try:
        m_score, m_ind = m_scorer.compute_score(gts_tok, res_tok)
    finally:
        if hasattr(m_scorer, 'lock') and m_scorer.lock.locked():
            m_scorer.lock.release()
        del m_scorer
    r_score, r_ind = Rouge().compute_score(gts_tok, res_tok)
    c_score, c_ind = Cider().compute_score(gts_tok, res_tok)

    det_metrics = {
        'BLEU-1': round(b_score[0] * 100, 2),
        'BLEU-2': round(b_score[1] * 100, 2),
        'BLEU-3': round(b_score[2] * 100, 2),
        'BLEU-4': round(b_score[3] * 100, 2),
        'METEOR': round(m_score * 100, 2),
        'ROUGE-L': round(r_score * 100, 2),
        'CIDEr': round(c_score * 100, 2)
    }

    # Load Oracle 1,000 report if available to include side-by-side comparison
    oracle_report_path = ROOT / "reports/VRSBENCH_QWEN_G2T_1000_EVAL_REPORT.json"
    oracle_metrics = {}
    oracle_caps_map = {}
    if oracle_report_path.exists():
        try:
            ora_data = json.loads(oracle_report_path.read_text(encoding='utf-8'))
            oracle_metrics = ora_data.get('metrics', {})
            for sp in ora_data.get('sample_predictions', []):
                oracle_caps_map[str(sp['question_id'])] = sp.get('predicted_caption', '')
        except Exception:
            pass

    key_order = list(gts_tok.keys())
    per_sample_scores = {}
    for i, k in enumerate(key_order):
        per_sample_scores[k] = {
            'BLEU-4': round(float(b_ind[3][i]) * 100, 2) if i < len(b_ind[3]) else 0.0,
            'METEOR': round(float(m_ind[i]) * 100, 2) if i < len(m_ind) else 0.0,
            'ROUGE-L': round(float(r_ind[i]) * 100, 2) if i < len(r_ind) else 0.0,
            'CIDEr': round(float(c_ind[i]) * 100, 2) if i < len(c_ind) else 0.0,
        }

    sample_results = []
    for s in test_samples:
        qid = s['question_id']
        sample_results.append({
            'image_id': s['image_id'],
            'question_id': qid,
            'ground_truth': s['ground_truth'],
            'oracle_scene_graph': s['oracle_graph'],
            'oracle_caption': oracle_caps_map.get(qid, ''),
            'detected_objects': s['detected_objects'],
            'predicted_scene_graph': s['predicted_graph'],
            'detector_caption': captions[qid],
            'detector_scores': per_sample_scores.get(qid, {})
        })

    out_data = {
        'task': 'End-to-End VRSBench 1,000-Sample Evaluation: Detector SG vs Oracle GT SG',
        'detector_checkpoint': str(ckpt_path.name),
        'adapter': args.adapter,
        'num_samples': len(test_samples),
        'detection_metrics': {
            'total_gt_boxes': total_gt_boxes,
            'total_pred_boxes': total_pred_boxes,
            'recall_at_50': det_recall,
            'precision_at_50': det_precision,
            'mean_max_iou': det_mean_iou
        },
        'detector_sg_caption_metrics': det_metrics,
        'oracle_sg_caption_metrics': oracle_metrics,
        'sample_predictions': sample_results
    }

    out_file = ROOT / args.out_report
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(out_data, f, indent=2)
    print(f"Saved End-to-End Detector-SG vs Oracle-SG report to {out_file}", flush=True)

if __name__ == '__main__':
    main()

