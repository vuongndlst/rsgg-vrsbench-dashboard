import os, signal, subprocess
res = subprocess.run(["nvidia-smi", "--id=0", "--query-compute-apps=pid,process_name", "--format=csv,noheader"], capture_output=True, text=True)
for line in res.stdout.strip().splitlines():
    parts = [p.strip() for p in line.split(',')]
    if len(parts) == 2 and 'python' in parts[1]:
        pid = int(parts[0])
        print(f"Killing old GPU 0 python process PID {pid}...")
        try:
            os.kill(pid, signal.SIGKILL)
            print(f"Killed PID {pid} successfully.")
        except Exception as e:
            print(f"Error killing {pid}: {e}")

