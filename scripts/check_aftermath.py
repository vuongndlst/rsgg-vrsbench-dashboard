import os, glob, time
from pathlib import Path

ROOT = Path.home() / 'rsg-former'
print("=== CHECK ARTIFACTS ===")
for p in sorted((ROOT / 'artifacts').glob('*vrsbench*')):
    mtime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(p.stat().st_mtime))
    print(p.name, p.stat().st_size, mtime)

print("=== CHECK REPORTS ===")
for p in sorted((ROOT / 'reports').glob('*VRSBENCH*')):
    mtime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(p.stat().st_mtime))
    print(p.name, p.stat().st_size, mtime)

rep = ROOT / 'reports' / 'VRSBENCH_DETECTOR_V2_TRAINING_METRICS.json'
if rep.exists():
    print("\nCONTENT OF VRSBENCH_DETECTOR_V2_TRAINING_METRICS.json:")
    print(rep.read_text(encoding='utf-8'))

print("\n=== CHECK ANY RECENT LOGS OR CORES ===")
for f in glob.glob('/tmp/*'):
    try:
        st = os.stat(f)
        if time.time() - st.st_mtime < 600:
            print("Recent /tmp:", f, st.st_size, time.strftime('%H:%M:%S', time.localtime(st.st_mtime)))
    except Exception:
        pass

