import subprocess

out = subprocess.getoutput("nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv")
print(out)
