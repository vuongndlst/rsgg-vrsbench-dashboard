import os

data_dir = os.path.join(os.path.expanduser("~"), "rsg-former/data/rsicd")
if os.path.exists(data_dir):
    for f in os.listdir(data_dir):
        fp = os.path.join(data_dir, f)
        if os.path.isfile(fp):
            print(f"File: {f} ({os.path.getsize(fp)/(1024*1024):.2f} MB)")
        elif os.path.isdir(fp):
            print(f"Dir: {f} ({len(os.listdir(fp))} items)")

