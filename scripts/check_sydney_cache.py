from pathlib import Path
rep = Path.home() / 'rsg-former' / 'reports'
for f in rep.glob("sydney*"):
    print(f.name, f.stat().st_size)

