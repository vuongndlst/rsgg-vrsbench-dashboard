import os, glob

base = os.path.expanduser('~/rsg-former')
if os.path.exists(base):
    os.chdir(base)
print("CWD:", os.getcwd())

print("--- Checkpoints ---")
if os.path.exists("checkpoints"):
    for root, dirs, files in os.walk("checkpoints"):
        for d in dirs:
            p = os.path.join(root, d)
            print("DIR:", p)
        for f in files:
            if f.endswith(('.pt', '.bin', '.safetensors', '.json')):
                p = os.path.join(root, f)
                print(f"FILE: {p} ({os.path.getsize(p)/(1024*1024):.2f} MB)")
else:
    print("No checkpoints dir")

print("\n--- Reports ---")
for f in sorted(glob.glob("reports/*")):
    print(f, f"({os.path.getsize(f)/(1024*1024):.2f} MB)")
