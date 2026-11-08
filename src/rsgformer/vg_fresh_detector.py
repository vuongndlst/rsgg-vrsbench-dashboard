"""Independent fresh detector builder: pinned native code, ImageNet backbone only.

Does not import the live v2 bridge or load any VG detector/SGG checkpoint.
The reference attention backend is a runtime difference, not native CUDA parity.
"""
from pathlib import Path
import hashlib
import importlib
import sys
import types

import torch
import torchvision

IMAGENET_SHA256 = '11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca'


def build_fresh_detector(root, seed=1729):
    root = Path(root)
    vendor = root / 'vendor/salience_sgg'
    # Use in a dedicated process; mixing two different native bridges is unsafe.
    if 'models' in sys.modules:
        raise RuntimeError('fresh_builder_requires_dedicated_process')
    sys.path.insert(0, str(vendor))
    package = types.ModuleType('models')
    package.__path__ = [str(vendor / 'models')]
    sys.modules['models'] = package
    sys.modules['MultiScaleDeformableAttention'] = types.ModuleType('MultiScaleDeformableAttention')
    attention = importlib.import_module('models.ops.functions.ms_deform_attn_func')

    def reference_apply(value, shapes, starts, locations, weights, step):
        return attention.ms_deform_attn_core_pytorch(
            value.float(), shapes, locations.float(), weights.float()).to(value.dtype)

    attention.MSDeformAttnFunction.apply = staticmethod(reference_apply)
    weights_path = root / 'weights/checkpoints/resnet50-11ad3fa6.pth'
    fingerprint = hashlib.sha256(weights_path.read_bytes()).hexdigest()
    if fingerprint != IMAGENET_SHA256:
        raise ValueError('imagenet_initialization_fingerprint_mismatch')
    state = torch.load(weights_path, map_location='cpu', weights_only=True)
    original = torchvision.models.resnet50
    backbone_calls = []

    def audited_resnet(*args, **kwargs):
        kwargs['weights'] = None
        backbone = original(*args, **kwargs)
        backbone.load_state_dict(state, strict=True)
        backbone_calls.append(True)
        return backbone

    import yaml
    from util.misc import ParserObject
    from models.dab_deformable_detr.deformable_detr import build
    config_path = vendor / 'configs/DABDeformableDETR_vg.yaml'
    config = yaml.safe_load(config_path.read_text())['model']
    assert config['backbone'] == 'resnet50' and config['num_classes'] == 150
    torch.manual_seed(seed)
    torchvision.models.resnet50 = audited_resnet
    try:
        model, criterion, postprocessors = build(ParserObject(config))
    finally:
        torchvision.models.resnet50 = original
    assert backbone_calls == [True]
    assert all(parameter.device.type == 'cpu' for parameter in model.parameters())
    provenance = {
        'seed': seed, 'initialization': 'ImageNet-1K V2 backbone; fresh transformer/query/class/box heads',
        'weights_sha256': fingerprint, 'vg_checkpoint_loaded': False,
        'attention_backend': 'pinned upstream PyTorch reference autograd function',
        'configuration_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest(),
        'parameters': sum(p.numel() for p in model.parameters()),
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'fully_contamination_free_claim': False,
    }
    return model, criterion, postprocessors, config, provenance
