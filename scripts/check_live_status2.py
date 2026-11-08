import os, glob, subprocess

print("CWD:", os.getcwd())
print("Reports in CWD:", glob.glob("reports/*"))
if os.path.exists("reports"):
    print("Files in reports:", os.listdir("reports"))

# check evaluate_finetuned_all.py process details
try:
    cmdline = open("/proc/937075/cmdline", "rb").read().decode("utf-8", "ignore").replace("\x00", " ")
    print("PID 937075 cmdline:", cmdline)
except Exception as e:
    print("PID 937075 error:", e)

# check any python processes under rsg-former
ps = subprocess.check_output(["ps", "-ef"], text=True)
for line in ps.splitlines():
    if "rsg-former" in line or "eval_" in line or "vrsbench" in line:
        print(line)

