from pathlib import Path
p = Path.home() / "EduSGG/launch_logs/overnight_finetune_and_eval_20261107_161530.log"
text = p.read_text(encoding="utf-8", errors="replace")
for line in text.splitlines()[110:140]:
    print(line)
