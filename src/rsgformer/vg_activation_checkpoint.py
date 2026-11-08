"""Instance-only checkpoint adapter for a separate fresh detector process.

Source/config files and parameters stay unchanged. This is a runtime change;
native-resolution memory and CUDA parity still require separate admission.
"""
from contextlib import contextmanager
from pathlib import Path
import hashlib
import types

import torch
from torch.utils.checkpoint import checkpoint

NATIVE_TRANSFORMER_SHA = '50d8d0d7ae5f2a363813757a89f322aa7e744b778e8bd93c393e1794fa28e247'


@contextmanager
def fresh_transformer_checkpointing(model, project_root):
    source = Path(project_root) / 'vendor/salience_sgg/models/dab_deformable_detr/deformable_transformer.py'
    if hashlib.sha256(source.read_bytes()).hexdigest() != NATIVE_TRANSFORMER_SHA:
        raise ValueError('native_transformer_source_changed')
    if getattr(model, '_fresh_checkpoint_context', False):
        raise RuntimeError('nested_checkpoint_context')
    groups = [model.transformer.encoder.layers, model.transformer.decoder.layers]
    expected = ['DeformableTransformerEncoderLayer', 'DeformableTransformerDecoderLayer']
    for layers, name in zip(groups, expected):
        if len(layers) != 6 or any(type(layer).__name__ != name for layer in layers):
            raise ValueError('unsupported_native_layer_layout')
    originals = []
    model._fresh_checkpoint_context = True
    try:
        for layers in groups:
            for layer in layers:
                original = layer.forward
                had_instance_forward = 'forward' in layer.__dict__
                previous_instance_forward = layer.__dict__.get('forward')

                def wrapped(instance, *args, _original=original, **kwargs):
                    if instance.training and torch.is_grad_enabled():
                        return checkpoint(_original, *args, use_reentrant=False,
                                          preserve_rng_state=True, **kwargs)
                    return _original(*args, **kwargs)

                originals.append((layer, had_instance_forward, previous_instance_forward))
                layer.forward = types.MethodType(wrapped, layer)
        yield {'encoder_layers': 6, 'decoder_layers': 6,
               'use_reentrant': False, 'preserve_rng_state': True}
    finally:
        for layer, had_forward, previous in reversed(originals):
            if had_forward:
                layer.forward = previous
            else:
                del layer.forward
        del model._fresh_checkpoint_context
