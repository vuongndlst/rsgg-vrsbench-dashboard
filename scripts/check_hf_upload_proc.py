import os, psutil
print("Checking processes on remote...")
for p in psutil.process_iter(['pid', 'name', 'cmdline']):
    try:
        cmd = " ".join(p.info['cmdline'] or [])
        if "upload_to_huggingface" in cmd or "remote_hf_upload" in cmd:
            print(f"PID: {p.info['pid']}, Name: {p.info['name']}, CPU: {p.cpu_percent()}%, Status: {p.status()}")
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass

