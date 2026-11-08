"""CPU-only audit of reused ImageNet initialization; never loads a VG detector."""
from pathlib import Path
from datetime import datetime, timezone
import ast
import hashlib
import json
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import torch
import torchvision
torch.set_num_threads(1)

root = Path.home() / 'rsg-former'
weights = root / 'weights/checkpoints/resnet50-11ad3fa6.pth'
source_path = root / 'vendor/salience_sgg/models/dab_deformable_detr/backbone.py'
assert weights.is_file()
digest = hashlib.sha256(weights.read_bytes()).hexdigest()
assert digest.startswith('11ad3fa6'), 'Expected torchvision ImageNet V2 content fingerprint'
enum = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
assert torchvision.models.ResNet50_Weights.DEFAULT == enum
assert enum.url == 'https://download.pytorch.org/models/resnet50-11ad3fa6.pth'
tree = ast.parse(source_path.read_text())
node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'FrozenBatchNorm2d')
namespace = {'torch': torch}
exec(compile(ast.Module(body=[node], type_ignores=[]), str(source_path), 'exec'), namespace)
model = torchvision.models.resnet50(weights=None, norm_layer=namespace['FrozenBatchNorm2d'])
state = torch.load(weights, map_location='cpu', weights_only=True)
model.load_state_dict(state, strict=True)
assert model.fc.out_features == 1000 and len(enum.meta['categories']) == 1000
out = {'status': 'PASS', 'utc': datetime.now(timezone.utc).isoformat(),
       'scope': 'CPU-only cached ImageNet backbone strict load into pinned native FrozenBatchNorm',
       'weights_path': '${REMOTE_HOME}/rsg-former/weights/checkpoints/resnet50-11ad3fa6.pth',
       'source_url': enum.url, 'pretraining': 'ImageNet-1K V2', 'weights_bytes': weights.stat().st_size,
       'weights_sha256': digest, 'source_backbone_sha256': hashlib.sha256(source_path.read_bytes()).hexdigest(),
       'torchvision_default_resolved': str(enum), 'strict_model_load': True,
       'vg_checkpoint_loaded': False, 'network_downloads': False, 'cuda_used': False,
       'new_training_started': False, 'fully_contamination_free_claim': False,
       'limitations': ['Known VG-held-out identity exclusions do not prove absence from ImageNet or Qwen pretraining corpora',
                      'Fresh detector query/class/regression heads must initialize without VG author weights',
                      'Fresh SGG relation prior must be recomputed from the clean training manifest, not reused from author VG prior',
                      'Detector criterion, augmentation, gradient/VRAM and throughput smoke remain required before launch']}
text = json.dumps(out, indent=2)
(root / 'reports/FRESH_BACKBONE_INITIALIZATION_AUDIT.json').write_text(text)
print(text)
