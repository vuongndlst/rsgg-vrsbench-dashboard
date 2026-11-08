import os

base = os.path.join(os.path.expanduser("~"), "rsg-former")
log_path = os.path.join(base, "launch_logs/overnight_finetune_and_eval_20261107_174649.log")
if os.path.exists(log_path):
    print("Found log at:", log_path)
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
        print("Total lines:", len(lines))
        print("--- Last 60 lines ---")
        for line in lines[-60:]:
            print(line, end="")
else:
    print("No log found at:", log_path)
    # Check what logs exist
    ll = os.path.join(base, "launch_logs")
    if os.path.exists(ll):
        print("Files in launch_logs:", os.listdir(ll))

