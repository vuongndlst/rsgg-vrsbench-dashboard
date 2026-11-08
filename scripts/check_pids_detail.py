import os, time

def check_proc(pid_name, pid):
    p = f"/proc/{pid}"
    if not os.path.exists(p):
        print(f"[{pid_name}] PID {pid} is NOT RUNNING!")
        return
    with open(f"{p}/stat") as f:
        stat = f.read().split()
        utime = int(stat[13])
        stime = int(stat[14])
        state = stat[2]
        print(f"[{pid_name}] PID {pid}: state={state}, utime={utime}, stime={stime}, total_cpu={utime+stime}")

check_proc("EVAL GPU 0", 310641)
check_proc("TRAIN GPU 1", 310116)

