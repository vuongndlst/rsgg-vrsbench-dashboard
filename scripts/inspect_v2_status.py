import os, json, time
from pathlib import Path

ROOT = Path.home() / 'rsg-former'
rep_file = ROOT / 'reports' / 'VRSBENCH_DETECTOR_V2_TRAINING_METRICS.json'
print("=== METRICS FILE CHECK ===")
if rep_file.exists():
    data = json.loads(rep_file.read_text(encoding='utf-8'))
    print("Epochs completed:", data.get('epochs_completed'))
    print("Best val mIoU:", data.get('best_val_mIoU'))
    print("History:")
    for h in data.get('epoch_history', []):
        print(f"  Epoch {h.get('epoch')}: Loss {h.get('loss_total_avg')}, Time {h.get('epoch_seconds', 0)/60:.1f}m, "
              f"mIoU {h.get('val_ag_mIoU')}%, Acc50 {h.get('val_ag_Acc50')}%, Acc70 {h.get('val_ag_Acc70')}% (Class-matched mIoU: {h.get('val_cl_mIoU')}%)")
else:
    print("Metrics file not found!")

print("\n=== CHECKPOINTS CHECK ===")
ckpts = sorted((ROOT / 'artifacts').glob('vrsbench_detector_v2*'))
for c in ckpts:
    mtime_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(c.stat().st_mtime))
    print(f"  {c.name}: {c.stat().st_size / (1024*1024):.1f} MB (modified {mtime_str})")

print("\n=== PID 12173 / GPU STATUS ===")
import subprocess
try:
    res = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,utilization.gpu', '--format=csv,noheader'], capture_output=True, text=True)
    print("NVIDIA-SMI:\n" + res.stdout)
except Exception as e:
    print("nvidia-smi err:", e)

p = Path('/proc/12173')
if p.exists():
    stat = p.joinpath('stat').read_text().split()
    utime = int(stat[13])
    stime = int(stat[14])
    state = stat[2]
    print(f"PID 12173: state={state}, utime={utime}, stime={stime}, total CPU jiffies={utime+stime}")
else:
    print("PID 12173 has finished or is not active.")

