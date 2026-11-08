import subprocess, time

t0 = time.time()
# Download 50MB to /dev/null
cmd = "curl -r 0-52428800 -s -w '%{speed_download} %{time_total}\n' -o /dev/null 'http://images.cocodataset.org/zips/train2014.zip'"
out = subprocess.getoutput(cmd)
print("Curl benchmark output (bytes/sec, seconds):", out)

