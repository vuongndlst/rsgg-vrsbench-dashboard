from pathlib import Path
for p in Path(".").rglob(".git"):
    print("Found git repo:", p)
for p in Path("..").glob("*/.git"):
    print("Found neighbor git repo:", p)

