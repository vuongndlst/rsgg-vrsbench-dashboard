import struct
from pathlib import Path

aria2_file = Path.home() / 'shared-assets/datasets/coco_karpathy/train2014.zip.aria2'
if not aria2_file.exists():
    print("No .aria2 file found!")
    exit(0)

data = aria2_file.read_bytes()
print(f"Total aria2 control file length: {len(data)} bytes")

# The aria2 format:
# 2 bytes: magic (0x0000)
# 4 bytes: version
# 4 bytes: length of extension (or similar)
# Let's inspect raw header
print("First 32 bytes hex:", data[:32].hex())

# Total expected pieces for 13510573713 with 1MB pieces is ~12885 pieces
# Let's count 1s in the data
bit_count = sum(bin(b).count('1') for b in data[32:])
print(f"Total 1 bits in payload (approx completed pieces): {bit_count} / 12885 (~{bit_count * 1048576 / (1024**3):.2f} GB)")

