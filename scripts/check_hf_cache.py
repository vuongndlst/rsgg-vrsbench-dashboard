import os, sys, glob
from pathlib import Path

print("=== CHECK HUGGINGFACE LOCKS ===")
cache_dir = Path.home() / ".cache/huggingface"
locks = list(cache_dir.rglob("*.lock"))
print(f"Found {len(locks)} lock files:")
for l in locks:
    print(f"  {l}")

print("\n=== CHECK HF MODELS IN CACHE ===")
hub_dir = cache_dir / "hub"
if hub_dir.exists():
    for d in hub_dir.glob("models--*"):
        print(f"  {d.name}")

