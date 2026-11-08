import os, subprocess, signal
from pathlib import Path

pids_to_kill = [492967, 496575, 497614, 497477, 500287, 502068]
for pid in pids_to_kill:
    try:
        os.kill(pid, signal.SIGTERM)
        print(f"Sent SIGTERM to {pid}")
    except ProcessLookupError:
        pass
    except Exception as e:
        print(f"Error killing {pid}: {e}")

# Clean up lock files and incomplete files in huggingface cache
hub_dir = Path.home() / ".cache/huggingface/hub"
for f in hub_dir.rglob("*.incomplete"):
    try:
        f.unlink()
        print(f"Removed incomplete file: {f.name}")
    except Exception as e:
        print(f"Error removing {f}: {e}")

locks_dir = hub_dir / ".locks"
if locks_dir.exists():
    for f in locks_dir.rglob("*.lock"):
        try:
            f.unlink()
            print(f"Removed lock file: {f.name}")
        except Exception as e:
            print(f"Error removing {f}: {e}")

print("Cleanup complete.")

