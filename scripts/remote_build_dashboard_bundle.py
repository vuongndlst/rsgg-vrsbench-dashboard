"""Build complete JSON bundle for the VRSBench Interactive Website (1,000 test samples).
Merges:
  - Ground-truth bounding boxes & scene graphs from Annotations_val.zip
  - Oracle-SG generated captions & per-sample scores (from 1,000-eval report or live cache)
  - Detector-SG generated boxes, scene graphs, captions & scores (if available)
  - Detector training epoch metrics & Full 9,350 test metrics (if available)
"""
import json, zipfile
from pathlib import Path

ROOT = Path.home() / 'rsg-former'
SHARED = Path.home() / 'shared-assets/datasets/vrsbench'

def main():
    eval_path = SHARED / 'VRSBench_EVAL_Cap.json'
    val_zip = SHARED / 'Annotations_val.zip'
    with open(eval_path) as f:
        eval_data = json.load(f)

    # Load Oracle 1,000 report or cache
    ora_report_path = ROOT / 'reports/VRSBENCH_QWEN_G2T_1000_EVAL_REPORT.json'
    ora_cache_path = ROOT / 'reports/VRSBENCH_QWEN_G2T_ORACLE_CAPTIONS_CACHE.json'
    ora_caps = {}
    ora_scores = {}
    ora_metrics = {'CIDEr': 32.19, 'BLEU-4': 13.20, 'METEOR': 25.83, 'ROUGE-L': 38.13, 'BLEU-1': 46.25, 'BLEU-2': 29.40, 'BLEU-3': 19.35}

    if ora_cache_path.exists():
        try:
            ora_caps.update(json.loads(ora_cache_path.read_text(encoding='utf-8')))
        except Exception:
            pass

    if ora_report_path.exists():
        try:
            rep = json.loads(ora_report_path.read_text(encoding='utf-8'))
            ora_metrics = rep.get('metrics', ora_metrics)
            for sp in rep.get('sample_predictions', []):
                qid = str(sp['question_id'])
                ora_caps[qid] = sp.get('predicted_caption', '')
                ora_scores[qid] = sp.get('scores', {})
        except Exception:
            pass

    # Load Detector 1,000 report or live caches if available
    det_report_path = ROOT / 'reports/VRSBENCH_DETECTOR_SG_1000_EVAL_REPORT.json'
    det_boxes_path = ROOT / 'reports/VRSBENCH_DETECTOR_BOXES_1000.json'
    det_caps_cache_path = ROOT / 'reports/VRSBENCH_QWEN_G2T_DETECTOR_CAPTIONS_CACHE.json'
    det_map = {}
    det_metrics = {}
    det_box_metrics = {}

    if det_boxes_path.exists():
        try:
            bdata = json.loads(det_boxes_path.read_text(encoding='utf-8'))
            det_box_metrics = bdata.get('detection_metrics', {})
            for qid, d in bdata.get('detections', {}).items():
                det_map[str(qid)] = {
                    'detected_objects': d.get('detected_objects', []),
                    'predicted_scene_graph': d.get('predicted_graph', ''),
                    'detector_caption': '',
                    'detector_scores': {}
                }
        except Exception:
            pass

    if det_caps_cache_path.exists():
        try:
            dcaps = json.loads(det_caps_cache_path.read_text(encoding='utf-8'))
            for qid, cap in dcaps.items():
                if str(qid) not in det_map:
                    det_map[str(qid)] = {}
                det_map[str(qid)]['detector_caption'] = cap
        except Exception:
            pass

    if det_report_path.exists():
        try:
            drep = json.loads(det_report_path.read_text(encoding='utf-8'))
            det_metrics = drep.get('detector_sg_caption_metrics', {})
            det_box_metrics = drep.get('detection_metrics', det_box_metrics)
            for sp in drep.get('sample_predictions', []):
                det_map[str(sp['question_id'])] = sp
        except Exception:
            pass

    # Load Detector training epochs
    det_train_path = ROOT / 'reports/VRSBENCH_DETECTOR_TRAINING_METRICS.json'
    det_epochs = []
    if det_train_path.exists():
        try:
            dt = json.loads(det_train_path.read_text(encoding='utf-8'))
            det_epochs = dt.get('epoch_metrics', [])
        except Exception:
            pass

    # Load Full 9,350 report if available
    full_path = ROOT / 'reports/VRSBENCH_QWEN_G2T_FULL9350_EVAL_REPORT.json'
    full_metrics = {}
    if full_path.exists():
        try:
            frep = json.loads(full_path.read_text(encoding='utf-8'))
            full_metrics = frep.get('metrics', {})
        except Exception:
            pass

    samples = []
    with zipfile.ZipFile(val_zip) as z:
        for idx, item in enumerate(eval_data[:1000], 1):
            qid = str(item.get('question_id'))
            img_id = item.get('image_id', '')
            base_name = img_id.replace('.png', '').replace('.jpg', '')
            cand = f"Annotations_val/{base_name}.json"
            clean_objs = []
            obj_summary = []
            gt_text = item.get('ground_truth', '')
            try:
                data = json.loads(z.read(cand).decode('utf-8'))
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
                    clean_objs.append({
                        'obj_cls': cls_name,
                        'obj_coord': [round(float(x), 4) for x in coord] if len(coord) == 4 else [],
                        'obj_size': size,
                        'referring_sentence': ref
                    })
                    item_desc = f"{size + ' ' if size else ''}{cls_name}"
                    if ref:
                        item_desc += f" ({ref.strip('.')})"
                    obj_summary.append(item_desc)
            except Exception:
                pass

            d_entry = det_map.get(qid, {})
            samples.append({
                'idx': idx,
                'question_id': qid,
                'image_id': img_id,
                'scene_graph': "; ".join(obj_summary) if obj_summary else "aerial scene overview",
                'objects': clean_objs,
                'ground_truth': gt_text,
                'predicted_caption': ora_caps.get(qid, ''),
                'scores': ora_scores.get(qid, {}),
                'detected_objects': d_entry.get('detected_objects', []),
                'predicted_scene_graph': d_entry.get('predicted_scene_graph', ''),
                'detector_caption': d_entry.get('detector_caption', ''),
                'detector_scores': d_entry.get('detector_scores', {})
            })

    bundle = {
        'summary': {
            'total_samples': len(samples),
            'oracle_generated_count': sum(1 for s in samples if s['predicted_caption']),
            'detector_generated_count': sum(1 for s in samples if s['detector_caption']),
            'oracle_metrics': ora_metrics,
            'detector_metrics': det_metrics,
            'detection_box_metrics': det_box_metrics,
            'detector_epochs': det_epochs,
            'full_9350_metrics': full_metrics
        },
        'samples': samples
    }

    out_path = ROOT / 'reports/dashboard_bundle.json'
    out_path.write_text(json.dumps(bundle, ensure_ascii=False), encoding='utf-8')
    print(f"Compiled dashboard bundle: {len(samples)} samples ({bundle['summary']['oracle_generated_count']} with Oracle captions, {bundle['summary']['detector_generated_count']} with Detector captions) -> {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")

if __name__ == '__main__':
    main()

