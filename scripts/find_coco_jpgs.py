import subprocess

cmd = "find /home/stackops -name 'COCO_train2014_*.jpg' 2>/dev/null | head -n 10"
res = subprocess.getoutput(cmd)
print("Find result for COCO_train2014:")
print(res if res else "None found")

cmd2 = "find /home/stackops -name 'COCO_val2014_*.jpg' 2>/dev/null | head -n 10"
res2 = subprocess.getoutput(cmd2)
print("Find result for COCO_val2014:")
print(res2 if res2 else "None found")

