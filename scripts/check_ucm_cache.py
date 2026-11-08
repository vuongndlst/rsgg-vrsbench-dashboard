from pathlib import Path
rep = Path.home() / 'rsg-former' / 'reports'
for f in rep.glob("*ucm*"):
    print(f.name, f.stat().st_size)
for f in rep.glob("*UCM*"):
    print(f.name, f.stat().st_size)
