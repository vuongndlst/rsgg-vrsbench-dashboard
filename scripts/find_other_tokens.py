import os
from pathlib import Path

candidates = [
    Path.home() / ".cache" / "huggingface" / "token",
    Path.home() / ".huggingface" / "token",
    Path.home() / ".cache" / "huggingface" / "stored_tokens",
    Path.home() / "rsg-former" / ".env",
    Path.home() / ".env"
]
for c in candidates:
    if c.exists():
        print(f"Found {c}:")
        content = c.read_text().strip()
        print(f"  Length: {len(content)}, Prefix: {content[:10]}...")

