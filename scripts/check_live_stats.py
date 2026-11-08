import os, time

p = "/proc/310116"
if os.path.exists(p):
    with open(f"{p}/stat") as f:
        stat = f.read().split()
        utime = int(stat[13])
        stime = int(stat[14])
        state = stat[2]
        print(f"TRAIN PID 310116: state={state}, utime={utime}, stime={stime}, total_cpu={utime+stime}")
else:
    print("TRAIN PID 310116 is NOT RUNNING!")

p2 = "/proc/310641"
if os.path.exists(p2):
    with open(f"{p2}/stat") as f:
        stat = f.read().split()
        utime = int(stat[13])
        stime = int(stat[14])
        state = stat[2]
        print(f"EVAL PID 310641: state={state}, utime={utime}, stime={stime}, total_cpu={utime+stime}")
else:
    print("EVAL PID 310641 is NOT RUNNING!")

