from pathlib import Path
for pid in [497614, 500287]:
    p = Path(f'/proc/{pid}/stack')
    if p.exists():
        try:
            print(f"PID {pid} stack:")
            print(p.read_text())
        except Exception as e:
            print(f"PID {pid} stack error: {e}")
    else:
        print(f"PID {pid} does not exist.")

