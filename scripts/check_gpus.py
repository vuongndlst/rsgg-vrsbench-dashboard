import subprocess
res = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_bus_id,pid,process_name,used_memory', '--format=csv,noheader'], capture_output=True, text=True)
print("Compute Apps:\n" + res.stdout)
