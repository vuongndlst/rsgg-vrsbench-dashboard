from pathlib import Path
p = Path('/proc/497614')
if p.exists():
    wchan = p.joinpath('wchan').read_text().strip() if p.joinpath('wchan').exists() else "unknown"
    stat = p.joinpath('stat').read_text().split()
    print("wchan:", wchan, "state:", stat[2])
    # check fd
    try:
        fds = list(p.joinpath('fd').iterdir())
        print("Open FDs:", len(fds))
        for fd in fds:
            try:
                print(" ", fd.name, "->", fd.resolve())
            except:
                pass
    except Exception as e:
        print("FD err:", e)

