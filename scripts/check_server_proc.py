import subprocess
print("=== NVIDIA-SMI ===")
try:
    print(subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,utilization.gpu', '--format=csv']).decode())
except Exception as e:
    print(f"Error nvidia-smi: {e}")

print("=== RSG-FORMER PROCESSES ===")
try:
    out = subprocess.check_output(['ps', '-ef']).decode()
    for line in out.splitlines():
        if any(k in line for k in ['python', 'qwen', 'ablation', 'java']):
            print(line[:120])
except Exception as e:
    print(f"Error ps: {e}")

