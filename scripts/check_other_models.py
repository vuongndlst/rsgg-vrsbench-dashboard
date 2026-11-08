import os
from pathlib import Path

p = Path.home() / "etechs/msvd-tkg-captioning/models"
for item in p.glob("*"):
    print(item.name, item.is_dir())

