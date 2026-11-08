from pathlib import Path
from transformers import AutoModelForCausalLM

model_path = str(Path.home() / 'models/git-large-coco')
model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
linear_modules = set()
for name, m in model.named_modules():
    if 'Linear' in m.__class__.__name__:
        linear_modules.add(name.split('.')[-1])
print("Linear module leaf names in GIT:")
for name in sorted(linear_modules):
    print(f"  {name}")

print("\nSample full module names:")
for i, (name, m) in enumerate(model.named_modules()):
    if 'attention' in name.lower() or 'proj' in name.lower() or 'q' in name.lower():
        if i < 30:
            print(f"  {name} ({m.__class__.__name__})")

