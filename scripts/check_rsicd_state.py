import os, glob, time
from pathlib import Path
ROOT = Path.expanduser(Path('~/rsg-former'))

print("Current time:", time.strftime('%X'))
for f in sorted((ROOT / 'reports').glob('*rsicd*')):
    st = f.stat()
    print(f"{f.name}: size={st.st_size} bytes, mtime={time.strftime('%X', time.localtime(st.st_mtime))}")

