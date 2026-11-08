import subprocess

print("Installing default-jre-headless...")
res = subprocess.run(['sudo', 'apt-get', 'update'], capture_output=True, text=True)
print("apt update rc:", res.returncode)

res_install = subprocess.run(['sudo', 'apt-get', 'install', '-y', 'default-jre-headless'], capture_output=True, text=True)
print("apt install rc:", res_install.returncode)
print("install tail:", res_install.stdout.splitlines()[-5:] if res_install.stdout else res_install.stderr[:300])

res_java = subprocess.run(['java', '-version'], capture_output=True, text=True)
print("java version rc:", res_java.returncode)
print("java stderr:", res_java.stderr[:200])

