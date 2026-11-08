import os, glob

home = os.path.expanduser("~")
print("Home:", home)
base = os.path.join(home, "rsg-former")
print("Base:", base, "Exists:", os.path.exists(base))

rep_dir = os.path.join(base, "reports")
if os.path.exists(rep_dir):
    print("\n--- Files in rsg-former/reports ---")
    for item in sorted(os.listdir(rep_dir)):
        p = os.path.join(rep_dir, item)
        size_kb = os.path.getsize(p) / 1024 if os.path.isfile(p) else 0
        print(f"  {item} ({size_kb:.1f} KB)")
else:
    print("Reports dir not found at", rep_dir)

# Check running processes and their logs
logs_dir = os.path.join(base, "launch_logs")
if os.path.exists(logs_dir):
    print("\n--- Files in launch_logs ---")
    for item in sorted(os.listdir(logs_dir)):
        p = os.path.join(logs_dir, item)
        print(f"  {item} ({os.path.getsize(p)} bytes)")

