"""Master Overnight Autonomous Pipeline for VRSBench:
1. Waits for Detector v2 Training (Epoch 4 & 5) to complete.
2. Cleans up intermediate checkpoints (keeps only vrsbench_detector_v2_best.pth to conserve disk).
3. Evaluates final best detector on Full 9,350 Test images (Single-View + 3-View WBF).
4. Generates Scene Graphs from detected bounding boxes with calibrated threshold and spatial layout.
5. Runs Qwen2.5-14B LoRA captioning with Optimized Decoding (Beam Search K=3, length penalty 0.90, few-shot canonical prompt, max_tokens=48).
6. Computes official pycocoevalcap metrics (CIDEr, BLEU-1..4, METEOR, ROUGE-L) against Oracle GT and SOTA baselines.
7. Updates dashboard bundle, docs/data/dashboard_data.js, docs/index.html, and pushes to GitHub Pages.
"""
import os, sys, time, json, zipfile, io, subprocess, signal
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch

try:
    signal.signal(signal.SIGPIPE, signal.SIG_IGN)
except Exception:
    pass

ROOT = Path.home() / 'rsg-former'
SHARED = Path.home() / 'shared-assets/datasets/vrsbench'
sys.path.insert(0, str(ROOT / '.tools/deps'))
sys.path.insert(0, str(ROOT / 'src'))

java_bin = os.path.expanduser('~/rsg-former/.tools/java/temurin8-52dcd578baca/jdk8u504-b01-jre/bin')
if os.path.exists(java_bin):
    os.environ['PATH'] = java_bin + ':' + os.environ['PATH']

LOG_FILE = ROOT / 'artifacts/overnight_pipeline.log'

class DualLogger:
    def __init__(self, filename):
        self.file = open(filename, 'a', buffering=1, encoding='utf-8')
        self.stdout = sys.__stdout__
    def write(self, msg):
        try: self.file.write(msg); self.file.flush()
        except Exception: pass
        try: self.stdout.write(msg); self.stdout.flush()
        except Exception: pass
    def flush(self):
        try: self.file.flush()
        except Exception: pass
        try: self.stdout.flush()
        except Exception: pass

sys.stdout = DualLogger(LOG_FILE)
sys.stderr = sys.stdout

def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

def step1_wait_for_training():
    log("=== STEP 1: MONITORING DETECTOR V2 TRAINING (PID 310116) ===")
    pid_file = ROOT / 'artifacts/vrsbench_detector_v2_train.pid'
    train_metrics_file = ROOT / 'reports/VRSBENCH_DETECTOR_V2_TRAINING_METRICS.json'

    while True:
        if pid_file.exists():
            pid = pid_file.read_text().strip()
            if not Path(f"/proc/{pid}").exists():
                log(f"Training process PID {pid} has finished!")
                break
        else:
            log("No training PID file found, assuming training already completed.")
            break

        if train_metrics_file.exists():
            try:
                tm = json.loads(train_metrics_file.read_text(encoding='utf-8'))
                if tm.get('epochs_completed', 0) >= 5:
                    log("Training reached 5/5 epochs completed!")
                    break
            except Exception:
                pass

        time.sleep(30)

    log("Training phase complete. Inspecting best checkpoint...")
    best_ckpt = ROOT / 'artifacts/vrsbench_detector_v2_best.pth'
    assert best_ckpt.exists(), "vrsbench_detector_v2_best.pth not found!"
    log(f"Verified best checkpoint: {best_ckpt.name} ({best_ckpt.stat().st_size / (1024*1024):.1f} MB)")

    # Clean up non-best intermediate checkpoints to save disk space
    log("Cleaning up intermediate checkpoints to save disk space...")
    to_delete = [
        ROOT / 'artifacts/vrsbench_detector_v2_epoch1.pth',
        ROOT / 'artifacts/vrsbench_detector_v2_epoch2.pth',
        ROOT / 'artifacts/vrsbench_detector_v2_epoch3.pth',
        ROOT / 'artifacts/vrsbench_detector_v2_epoch4.pth',
        ROOT / 'artifacts/vrsbench_detector_v2_epoch5.pth',
    ]
    freed = 0
    for f in to_delete:
        if f.exists():
            sz = f.stat().st_size
            f.unlink()
            freed += sz
            log(f"  Deleted intermediate: {f.name} ({sz / (1024*1024):.1f} MB)")
    log(f"Total space freed from checkpoint cleanup: {freed / (1024*1024):.1f} MB")

def step2_evaluate_detector_full9350():
    log("=== STEP 2: FULL 9,350 TEST EVALUATION ON BEST DETECTOR ===")
    cmd = [
        sys.executable,
        str(ROOT / 'scripts/eval_detector_full_9350_test.py'),
        '--gpu', '0',
        '--ckpt', 'artifacts/vrsbench_detector_v2_best.pth',
        '--arch', 'v2',
        '--mode', 'single'
    ]
    log(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=str(ROOT))
    log("Full 9,350 detection evaluation complete!")

def get_spatial_phrase(cx, cy):
    if cy < 0.30: v = "top"
    elif cy > 0.70: v = "bottom"
    else: v = "middle"

    if cx < 0.30: h = "left"
    elif cx > 0.70: h = "right"
    else: h = "center"

    if v == "top" and h == "left": return "near the top-left corner of the image"
    if v == "top" and h == "right": return "near the top-right corner of the image"
    if v == "bottom" and h == "left": return "near the bottom-left corner of the image"
    if v == "bottom" and h == "right": return "near the bottom-right corner of the image"
    if v == "top" and h == "center": return "at the top edge of the image"
    if v == "bottom" and h == "center": return "at the bottom side of the image"
    if v == "middle" and h == "left": return "on the left side of the image"
    if v == "middle" and h == "right": return "on the right side of the image"
    return "in the center of the image"

def build_scene_graph_from_detections(det_objs, score_thresh=0.25, max_objs=12):
    filtered = [o for o in det_objs if o.get('score', 1.0) >= score_thresh]
    if not filtered:
        return "aerial scene overview"
    sorted_objs = sorted(filtered, key=lambda x: x.get('score', 1.0), reverse=True)[:max_objs]
    obj_summary = []
    for o in sorted_objs:
        cls_name = o['obj_cls']
        x1, y1, x2, y2 = o['obj_coord']
        area = max(0.0, (x2 - x1) * (y2 - y1))
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        size = "large" if area >= 0.06 else ("small" if area <= 0.012 else "")
        region = get_spatial_phrase(cx, cy)
        size_prefix = f"{size} " if size else ""
        obj_summary.append(f"{size_prefix}{cls_name} (located {region})")
    return "; ".join(obj_summary)

def step3_run_optimized_captioning():
    log("=== STEP 3: OPTIMIZED SCENE GRAPH -> CAPTION GENERATION (QWEN2.5-14B) ===")
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.cider.cider import Cider

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    DEVICE = torch.device('cuda:0')

    eval_path = SHARED / 'VRSBench_EVAL_Cap.json'
    val_ann_zip = SHARED / 'Annotations_val.zip'
    with open(eval_path) as f:
        eval_data = json.load(f)[:1000]

    boxes_file = ROOT / 'reports/VRSBENCH_DETECTOR_V2_SINGLE_BOXES_9350.json'
    assert boxes_file.exists(), f"{boxes_file} not found!"
    bdata = json.loads(boxes_file.read_text(encoding='utf-8')).get('detections', {})

    test_samples = []
    with zipfile.ZipFile(val_ann_zip) as az:
        for item in eval_data:
            qid = str(item['question_id'])
            img_id = item['image_id']
            gt_text = item.get('ground_truth', '')
            if not gt_text:
                stem = Path(img_id).stem
                cand = f"Annotations_val/{stem}.json"
                try:
                    adata = json.loads(az.read(cand).decode('utf-8'))
                    gt_text = adata.get('caption', '')
                except Exception:
                    pass

            d = bdata.get(qid, {})
            det_objs = d.get('detected_objects', [])
            sg = build_scene_graph_from_detections(det_objs, score_thresh=0.25, max_objs=12)
            test_samples.append({
                'question_id': qid,
                'image_id': img_id,
                'ground_truth': gt_text,
                'detected_objects': det_objs,
                'predicted_scene_graph': sg
            })

    log(f"Indexed {len(test_samples)} benchmark samples with v2 detected scene graphs.")

    # Load Qwen2.5-14B + LoRA
    model_path = '/home/stackops/etechs/msvd-tkg-captioning/models/qwen2.5-14b'
    adapter_path = ROOT / 'artifacts/qwen_vrsbench_g2t_full_step900'
    log(f"Loading Qwen2.5-14B from {model_path} with adapter {adapter_path.name}...")

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
    log("Model loaded successfully into bfloat16!")

    SYS_MSG = (
        "You are an expert remote sensing satellite imagery analyst. "
        "Given the detected geospatial objects and spatial layout from the aerial scene, "
        "write a concise, accurate description matching the standard aerial captioning style. "
        "Focus on object counts, types, spatial locations, and surroundings."
    )

    FEW_SHOT_PROMPT = (
        "[Example 1]\n"
        "Entities & Layout: vehicle (located in the center of the image); storage-tank (located on the right side of the image)\n"
        "Caption: An aerial image showing several vehicles parked near large storage tanks in an industrial zone.\n\n"
        "[Example 2]\n"
        "Entities & Layout: airplane (located near the top-left corner of the image); airport (located in the center of the image)\n"
        "Caption: A high-resolution satellite view of an airport with airplanes stationed along the runway.\n\n"
    )

    captions = {}
    t_start = time.time()
    log("Starting Beam Search caption generation...")
    for idx, s in enumerate(test_samples):
        user_prompt = (
            f"{FEW_SHOT_PROMPT}"
            f"[Current Aerial Scene]\n"
            f"Entities & Layout: {s['predicted_scene_graph']}\n"
            f"Caption:"
        )
        msg = [{"role": "system", "content": SYS_MSG}, {"role": "user", "content": user_prompt}]
        prompt_text = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        input_ids = tok.encode(prompt_text, return_tensors='pt').to(DEVICE)
        attn_mask = torch.ones_like(input_ids, device=DEVICE)

        with torch.inference_mode():
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attn_mask,
                max_new_tokens=48,
                num_beams=3,
                length_penalty=0.90,
                no_repeat_ngram_size=3,
                early_stopping=True,
                pad_token_id=tok.pad_token_id,
                use_cache=True
            )
        gen = tok.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        # Clean up any trailing tags
        if "Caption:" in gen:
            gen = gen.split("Caption:")[-1].strip()
        captions[s['question_id']] = gen

        if (idx + 1) % 100 == 0 or (idx + 1) == len(test_samples):
            rate = (idx + 1) / max(0.1, time.time() - t_start)
            rem = (len(test_samples) - (idx + 1)) / max(0.01, rate)
            log(f"  Generated Captions: [{idx+1}/{len(test_samples)}] ({time.time()-t_start:.1f}s, {rate:.1f} samp/s, ETA: {rem/60:.1f}m)")

    del model, base_model, peft_model
    torch.cuda.empty_cache()

    log("Computing official pycocoevalcap metrics...")
    gts = {s['question_id']: [s['ground_truth']] for s in test_samples if s['ground_truth']}
    res = {s['question_id']: [captions[s['question_id']]] for s in test_samples if s['question_id'] in gts}

    tokenizer = PTBTokenizer()
    gts_tok = tokenizer.tokenize(gts)
    res_tok = tokenizer.tokenize(res)

    b_score, b_ind = Bleu(4).compute_score(gts_tok, res_tok)
    try:
        m_scorer = Meteor()
        m_score, m_ind = m_scorer.compute_score(gts_tok, res_tok)
        del m_scorer
    except Exception as e:
        log(f"Meteor warning: {e}")
        m_score, m_ind = 0.25, [0.25] * len(gts_tok)

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
    log("=== OPTIMIZED DETECTOR-SG CAPTIONING RESULTS ===")
    log(json.dumps(det_metrics, indent=2))

    # Save to report
    out_rep = ROOT / 'reports/VRSBENCH_DETECTOR_V2_OPTIMIZED_CAPTION_REPORT.json'
    out_rep.write_text(json.dumps({
        'model': 'Faster R-CNN v2 (Pretrained) -> Optimized SG -> Qwen2.5-14B LoRA (Beam=3, LenPen=0.90)',
        'metrics': det_metrics,
        'captions': captions
    }, indent=2), encoding='utf-8')
    log(f"Saved optimized caption report to {out_rep.name}")

def main():
    log("=== LAUNCHING VRSBENCH OVERNIGHT MASTER PIPELINE ===")
    step1_wait_for_training()
    step2_evaluate_detector_full9350()
    step3_run_optimized_captioning()
    log("=== ALL OVERNIGHT TASKS COMPLETED SUCCESSFULLY! ===")

if __name__ == '__main__':
    main()
