import os, sys, time, json, subprocess
from pathlib import Path

ROOT = Path.home() / 'rsg-former'

print("=== 1. GPU UTILIZATION ===")
try:
    res = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,utilization.gpu', '--format=csv,noheader'], capture_output=True, text=True)
    print(res.stdout.strip())
except Exception as e:
    print(e)

print("\n=== 2. EVALUATION STATUS (GPU 0 - PID in reports/eval_v2_full9350.pid) ===")
eval_pid_file = ROOT / 'reports/eval_v2_full9350.pid'
eval_log = ROOT / 'reports/eval_v2_full9350.log'
if eval_pid_file.exists():
    pid = eval_pid_file.read_text().strip()
    active = Path(f"/proc/{pid}").exists()
    print(f"Evaluation PID: {pid} (Active: {active})")
if eval_log.exists():
    lines = eval_log.read_text(encoding='utf-8', errors='replace').splitlines()
    print(f"Eval log tail (total lines: {len(lines)}):")
    for l in lines[-10:]:
        print("  " + l)

eval_report = ROOT / 'reports/VRSBENCH_DETECTOR_FULL9350_V2_SINGLE_REPORT.json'
if eval_report.exists():
    print("\n*** EVALUATION REPORT FOUND! ***")
    rep_data = json.loads(eval_report.read_text(encoding='utf-8'))
    print(json.dumps({k: v for k, v in rep_data.items() if k != 'ap_metrics'}, indent=2))
    print("mAP metrics:", json.dumps({k: v for k, v in rep_data.get('ap_metrics', {}).items() if k != 'per_class_ap50'}, indent=2))

print("\n=== 3. TRAINING STATUS (GPU 1 - PID in artifacts/vrsbench_detector_v2_train.pid) ===")
train_pid_file = ROOT / 'artifacts/vrsbench_detector_v2_train.pid'
train_log = ROOT / 'artifacts/vrsbench_detector_v2_train.log'
if train_pid_file.exists():
    pid = train_pid_file.read_text().strip()
    active = Path(f"/proc/{pid}").exists()
    print(f"Training PID: {pid} (Active: {active})")
if train_log.exists():
    lines = train_log.read_text(encoding='utf-8', errors='replace').splitlines()
    print(f"Train log tail (total lines: {len(lines)}):")
    for l in lines[-10:]:
        print("  " + l)

train_metrics = ROOT / 'reports/VRSBENCH_DETECTOR_V2_TRAINING_METRICS.json'
if train_metrics.exists():
    try:
        tdata = json.loads(train_metrics.read_text(encoding='utf-8'))
        print(f"Epochs completed: {tdata.get('epochs_completed')} / {tdata.get('total_epochs')}")
        print(f"Best val mIoU: {tdata.get('best_val_mIoU')}%")
    except Exception:
        pass

