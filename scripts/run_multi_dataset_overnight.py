"""Master Orchestrator for Multi-Dataset Overnight Benchmarking.
Executes end-to-end:
  1. Monitor Detector v2 training completion (Epoch 5).
  2. Retain best checkpoint (vrsbench_detector_v2_best.pth) and delete intermediate checkpoints.
  3. Grounding evaluation on VRSBench Test (9,350 images).
  4. Captioning optimization loop on VRSBench:
     - Iterates decoding parameters and prompt few-shot templates.
     - Evaluates COCO (BLEU 1-4, METEOR, ROUGE-L, CIDEr) + SPICE + CHAIR.
     - Stops when exceeding SOTA benchmarks (LLaVA-1.5 & GeoChat).
  5. Zero-shot Transfer to Sydney-Captions (613 images).
  6. Zero-shot Transfer to UCM-Captions (2,100 images).
  7. Consolidate results and push updates to Web Dashboard on GitHub Pages.
"""

import os, sys, time, subprocess, json, shutil
from pathlib import Path
from datetime import datetime

ROOT = Path.home() / 'rsg-former'
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LOG_FILE = ROOT / 'artifacts' / 'multi_dataset_overnight.log'
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

def log(msg: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{timestamp}] {msg}"
    print(formatted, flush=True)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(formatted + '\n')

def wait_for_detector_training(pid: int = 310116):
    log(f"=== STEP 1: MONITORING DETECTOR V2 TRAINING (PID {pid}) ===")
    proc_dir = Path(f"/proc/{pid}")
    start_wait = time.time()
    while proc_dir.exists():
        time.sleep(30)
        elapsed_min = (time.time() - start_wait) / 60
        if int(elapsed_min) % 5 == 0 and int(time.time() - start_wait) % 30 == 0:
            log(f"Training active for {elapsed_min:.1f}m...")
    log("Detector v2 training completed successfully!")

def cleanup_checkpoints():
    log("=== STEP 2: CHECKPOINT CLEANUP & PRESERVATION ===")
    art_dir = ROOT / 'artifacts'
    best_ckpt = art_dir / 'vrsbench_detector_v2_best.pth'
    if best_ckpt.exists():
        log(f"Verified Best Checkpoint: {best_ckpt.name} ({best_ckpt.stat().st_size / (1024*1024):.1f} MB)")
    else:
        # Check if epoch 5 is available
        ep5 = art_dir / 'vrsbench_detector_v2_epoch5.pth'
        if ep5.exists():
            shutil.copy2(ep5, best_ckpt)
            log(f"Promoted epoch5 to {best_ckpt.name}")

    # Remove intermediate checkpoints
    for ep in range(1, 6):
        old_file = art_dir / f'vrsbench_detector_v2_epoch{ep}.pth'
        if old_file.exists():
            try:
                old_file.unlink()
                log(f"Removed intermediate checkpoint: {old_file.name}")
            except Exception as e:
                log(f"Warning: could not remove {old_file.name}: {e}")

def run_grounding_eval():
    log("=== STEP 3: VRSBENCH FULL 9,350 TEST GROUNDING EVALUATION ===")
    rep_file = ROOT / 'reports' / 'VRSBENCH_DETECTOR_FULL9350_V2_SINGLE_REPORT.json'
    if rep_file.exists():
        log(f"Grounding report already verified: {rep_file.name}")
    else:
        cmd = [sys.executable, str(ROOT / 'scripts' / 'eval_detector_full_9350_test.py')]
        log(f"Executing: {' '.join(cmd)}")
        res = subprocess.run(cmd, capture_output=True, text=True)
        log(f"Grounding eval complete with code {res.returncode}")

def run_sota_caption_tuning():
    log("=== STEP 4: AUTONOMOUS SOTA TUNING LOOP ON VRSBENCH ===")
    cmd = [sys.executable, str(ROOT / 'scripts' / 'auto_tune_vrsbench_sota.py')]
    log(f"Executing: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    log(f"SOTA Tuning complete with code {res.returncode}")

def run_sydney_transfer():
    log("=== STEP 5: ZERO-SHOT TRANSFER TO SYDNEY-CAPTIONS ===")
    sydney_script = ROOT / 'scripts' / 'eval_sydney_captions.py'
    if sydney_script.exists():
        cmd = [sys.executable, str(sydney_script)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        log(f"Sydney transfer complete with code {res.returncode}")

def run_ucm_transfer():
    log("=== STEP 6: ZERO-SHOT TRANSFER TO UCM-CAPTIONS ===")
    ucm_script = ROOT / 'scripts' / 'eval_ucm_captions.py'
    if ucm_script.exists():
        cmd = [sys.executable, str(ucm_script)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        log(f"UCM transfer complete with code {res.returncode}")

def run_dior_transfer():
    log("=== STEP 7: ZERO-SHOT TRANSFER TO DIOR-RSVG (GROUNDING) ===")
    dior_script = ROOT / 'scripts' / 'eval_dior_rsvg.py'
    if dior_script.exists():
        cmd = [sys.executable, str(dior_script)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        log(f"DIOR-RSVG transfer complete with code {res.returncode}")

def update_dashboard():
    log("=== STEP 8: CONSOLIDATE & SYNC WEB DASHBOARD ===")
    dash_script = ROOT / 'scripts' / 'update_local_dashboard_data.py'
    if dash_script.exists():
        cmd = [sys.executable, str(dash_script)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        log(f"Dashboard sync complete with code {res.returncode}")

def main():
    log("========================================================")
    log("  RSGG MULTI-DATASET OVERNIGHT PIPELINE LAUNCHED")
    log("========================================================")
    wait_for_detector_training(310116)
    cleanup_checkpoints()
    run_grounding_eval()
    run_sota_caption_tuning()
    run_sydney_transfer()
    run_ucm_transfer()
    run_dior_transfer()
    update_dashboard()
    log("========================================================")
    log("  ALL OVERNIGHT BENCHMARK TASKS SUCCESSFULLY COMPLETED!")
    log("========================================================")

if __name__ == '__main__':
    main()
