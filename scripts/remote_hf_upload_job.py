import sys, os
from pathlib import Path

# Add remote project root to sys.path
candidates = [
    Path.home() / "rsg-former",
    Path.cwd(),
    Path("/home") / os.environ.get("USER", "") / "rsg-former"
]
for c in candidates:
    if c.exists() and str(c) not in sys.path:
        sys.path.insert(0, str(c))

from scripts.upload_to_huggingface import main

# Pass arguments reading token from environment
token = os.environ.get("HF_TOKEN")
sys.argv = [
    "upload_to_huggingface.py",
    "--repo-id", "ndvuongsp/rsgg-vrsbench-models"
]
if token:
    sys.argv.extend(["--token", token])

if __name__ == "__main__":
    main()
