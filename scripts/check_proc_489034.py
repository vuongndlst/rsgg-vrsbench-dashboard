from pathlib import Path
p = Path('/proc/489034')
print("PID 489034 alive:", p.exists())
if p.exists():
    stat = p.joinpath('stat').read_text().split()
    print("State:", stat[2], "CPU jiffies:", int(stat[13])+int(stat[14]))

