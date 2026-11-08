import os, glob

project_dir = "${REMOTE_HOME}/rsg-former"
print("Project dir exists:", os.path.exists(project_dir))
if os.path.exists(project_dir):
    rep_dir = os.path.join(project_dir, "reports")
    print("Reports dir exists:", os.path.exists(rep_dir))
    if os.path.exists(rep_dir):
        for item in sorted(os.listdir(rep_dir)):
            full = os.path.join(rep_dir, item)
            size = os.path.getsize(full) if os.path.isfile(full) else 0
            mtime = os.path.getmtime(full)
            print(f"{item} ({size/1024:.1f} KB, mtime={mtime})")

# Check log file
log_path = os.path.join(project_dir, "launch_logs/overnight_finetune_and_eval_20261107_174649.log")
if not os.path.exists(log_path):
    # check in ~
    log_path = "launch_logs/overnight_finetune_and_eval_20261107_174649.log"

print("\nLog file exists:", os.path.exists(log_path))
if os.path.exists(log_path):
    print(f"Log size: {os.path.getsize(log_path)} bytes")
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
        print("Total lines:", len(lines))
        print("--- Last 50 lines ---")
        for line in lines[-50:]:
            print(line, end="")

