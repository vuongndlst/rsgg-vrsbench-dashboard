import os

try:
    cwd = os.readlink("/proc/937075/cwd")
    print("PID 937075 CWD:", cwd)
    log_cand = os.path.join(cwd, "launch_logs/overnight_finetune_and_eval_20261107_174649.log")
    if os.path.exists(log_cand):
        print("Found log at:", log_cand)
        with open(log_cand, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
            print("Total lines:", len(lines))
            print("--- Last 50 lines ---")
            for line in lines[-50:]:
                print(line, end="")
except Exception as e:
    print("Error:", e)

