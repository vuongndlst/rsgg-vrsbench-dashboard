import shutil, subprocess
from pathlib import Path

train_dir = Path.home() / 'shared-assets/datasets/coco_karpathy/images/train2014'
print(f"Target to delete: {train_dir}")
if train_dir.exists():
    print(f"Deleting {train_dir}...")
    shutil.rmtree(train_dir)
    print("Deleted successfully!")
else:
    print("train2014 does not exist!")

# Also check for any train2014 zip
train_zip = Path.home() / 'shared-assets/datasets/coco_karpathy/train2014.zip'
if train_zip.exists():
    print(f"Deleting {train_zip}...")
    train_zip.unlink()

res = subprocess.run(['df', '-h', '/'], capture_output=True, text=True)
print("\nUpdated Server Disk Space:\n" + res.stdout)

