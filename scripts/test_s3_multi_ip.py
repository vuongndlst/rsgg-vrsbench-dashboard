import subprocess, time

ips = [
  "16.15.199.6",
  "54.231.204.90",
  "54.231.230.250",
  "16.15.161.166",
  "52.217.82.208",
  "52.217.229.82",
  "16.15.213.135",
  "16.15.191.153"
]

for ip in ips[:3]:
    cmd = f"curl -s -r 0-5242880 --resolve images.cocodataset.org:80:{ip} -w '%{{speed_download}} %{{time_total}}\n' -o /dev/null 'http://images.cocodataset.org/zips/train2014.zip'"
    t0 = time.time()
    out = subprocess.getoutput(cmd)
    print(f"IP {ip}: {out.strip()} (elapsed: {time.time()-t0:.2f}s)")

