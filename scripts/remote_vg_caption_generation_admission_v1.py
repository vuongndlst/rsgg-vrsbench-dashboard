"""Read-only aggregate storage/GPU/RAM admission for the new caption controller."""
from pathlib import Path
from datetime import datetime,timezone
import json,shutil,subprocess
root=Path.home()/'rsg-former';total=0;seen=set()
for p in (root/'artifacts').rglob('*'):
 if p.is_symlink() or not p.is_file():continue
 st=p.stat();key=(st.st_dev,st.st_ino)
 if key not in seen:total+=st.st_size;seen.add(key)
free=shutil.disk_usage(root).free
gpu=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0])*1024**2
ram=int(next(x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')).split()[1])*1024
result={'utc':datetime.now(timezone.utc).isoformat(),'aggregate_artifacts_bytes_deduplicated_inodes':total,'aggregate_cap_bytes':40*1024**3,
 'disk_free_bytes':free,'GPU_free_bytes':gpu,'RAM_available_bytes':ram,'admitted':total<=40*1024**3 and free>=17*1024**3 and gpu>=38*1024**3 and ram>=14*1024**3,'TEST_accessed':False}
print(json.dumps(result))
