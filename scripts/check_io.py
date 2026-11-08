import os, time

p = "/proc/310641"
if os.path.exists(p):
    with open(f"{p}/io") as f:
        print("PID 310641 (EVAL) IO stats:\n" + f.read())

p2 = "/proc/310116"
if os.path.exists(p2):
    with open(f"{p2}/io") as f:
        print("PID 310116 (TRAIN) IO stats:\n" + f.read())

