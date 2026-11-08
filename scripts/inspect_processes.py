import os, subprocess, sys

def run(cmd):
    return subprocess.getoutput(cmd)

print("=== PID 497477 (run_multi_dataset_overnight.py) ===")
print(run("ps -f -p 497477"))

print("\n=== PID 497614 (eval_sydney_captions.py) ===")
print(run("ps -f -p 497614"))
print(run("lsof -p 497614 | grep -E 'models|cuda|log|json' | tail -20"))

print("\n=== Check PID 497614 stack or status ===")
print(run("cat /proc/497614/status | grep -E 'State|VmRSS|Threads'"))

print("\n=== Any recent logs in reports or rsg-former? ===")
print(run("ls -lt ~/rsg-former/reports | head -15"))

print("\n=== Check PID 497477 log / file descriptors ===")
print(run("ls -l /proc/497477/fd"))
print(run("ls -l /proc/497614/fd"))

