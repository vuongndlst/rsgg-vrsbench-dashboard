import os, subprocess, json, glob
from pathlib import Path

def run_cmd(cmd):
    return subprocess.getoutput(cmd)

print("=== DISK SPACE ===")
print(run_cmd("df -h /"))

print("\n=== GPU STATUS ===")
print(run_cmd("nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader"))

print("\n=== RELEVANT PROCESSES ===")
print(run_cmd("ps -eo pid,user,etime,%cpu,%mem,args | grep -E 'python|eval|auto_tune|run_multi' | grep -v grep"))

print("\n=== REPORTS DIRECTORY ===")
reports_dir = Path.home() / "rsg-former/reports"
if reports_dir.exists():
    for f in sorted(reports_dir.glob("*")):
        print(f"  {f.name} ({f.stat().st_size} bytes)")

print("\n=== MASTER OVERNIGHT LOG ===")
log_path = Path.home() / "rsg-former/reports/overnight_pipeline.log"
if log_path.exists():
    print(f"Size: {log_path.stat().st_size} bytes")
    print("Last 30 lines:")
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    for l in lines[-30:]:
        print(l)
else:
    print("overnight_pipeline.log does not exist yet.")

