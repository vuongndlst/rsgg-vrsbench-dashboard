import subprocess
res = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader'], capture_output=True, text=True)
print("GPU compute apps:\n" + res.stdout)

