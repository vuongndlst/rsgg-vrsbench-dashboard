from pathlib import Path
shared = Path.home() / 'shared-assets/datasets/coco_karpathy'
for p in shared.rglob('*.json'):
    print(p.relative_to(shared), f"({p.stat().st_size / (1024**2):.1f} MB)")

