"""Synthetic FP32 output/loss/full-gradient parity; no GPU/dataset training."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from pathlib import Path
from datetime import datetime, timezone
from contextlib import nullcontext
import gc
import hashlib
import json
import resource
import shutil
import signal
import sys
import time

root = Path.home() / 'rsg-former'
folder = root / 'artifacts/caption_readiness/fresh_checkpoint_cpu_probe_v1'
assert not folder.exists(), 'preserve_probe_version_no_hidden_retry'
assert shutil.disk_usage(root).free > 16 * 1024 ** 3
signal.alarm(120)
os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
sys.path.insert(0, str(root / 'src'))
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
from rsgformer.vg_fresh_detector import build_fresh_detector
from rsgformer.vg_activation_checkpoint import fresh_transformer_checkpointing

cpu = json.loads((root / 'reports/FRESH_DETECTOR_CPU_PROBE.json').read_text())
assert cpu['status'] == 'PASS'
hashes = dict(cpu['source_hashes'])
for name, expected in hashes.items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
for name in ['src/rsgformer/vg_activation_checkpoint.py', 'scripts/remote_fresh_checkpoint_cpu_probe.py']:
    hashes[name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
protocol = {'utc': datetime.now(timezone.utc).isoformat(), 'source_hashes': hashes,
            'scope': 'invented 128x160 image, same model parameters, native train-mode dropout, native criterion',
            'seed_initialization': 1729, 'seed_input': 19283, 'seed_dropout': 7173,
            'precision': 'FP32', 'gradient_rtol': 1e-5, 'gradient_atol': 1e-6,
            'checkpoint': 'six encoder + six decoder layers; nonreentrant; preserve RNG',
            'resource': {'cpu_threads': 1, 'alarm_seconds': 120, 'peak_rss_gate_gib': 4},
            'optimizer_steps': 0, 'benchmark_data_accessed': False, 'cuda_used': False}
folder.mkdir()
(folder / 'preregistration.json').write_text(json.dumps(protocol, indent=2))
started = time.monotonic()
report = {'status': 'FAILED', 'protocol': protocol,
          'preregistration_sha256': hashlib.sha256((folder / 'preregistration.json').read_bytes()).hexdigest(),
          'cuda_used': False, 'dataset_training_launched': False, 'benchmark_test_scored': False,
          'live_v2_modified': False, 'native_resolution_resource_gate_passed': False,
          'checkpoint_saved': False, 'optimizer_step_executed': False}
try:
    model, criterion, post, config, provenance = build_fresh_detector(root)
    from util.misc import nested_tensor_from_tensor_list
    model.train()
    assert any(isinstance(m, torch.nn.Dropout) and m.p > 0 for m in model.modules())
    torch.manual_seed(19283)
    samples = nested_tensor_from_tensor_list([torch.randn(3, 128, 160)])
    targets = [{'labels': torch.tensor([0, 74, 149]),
                'boxes': torch.tensor([[.2, .3, .15, .2], [.55, .5, .3, .25], [.8, .75, .12, .15]])}]
    baseline_outputs = None
    baseline_losses = None
    baseline_grads = None
    timings, saved_bytes = {}, {}
    max_output_error = max_gradient_error = max_loss_error = 0.0
    for name, context in [('native', nullcontext()),
                          ('checkpointed', fresh_transformer_checkpointing(model, root))]:
        model.zero_grad(set_to_none=True)
        torch.manual_seed(7173)
        saved = [0]

        def pack(tensor):
            saved[0] += tensor.numel() * tensor.element_size()
            return tensor

        t = time.monotonic()
        with context:
            with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
                outputs = model(samples)
                losses = criterion(outputs, targets)
                loss = sum(losses[key] * weight for key, weight in criterion.weight_dict.items() if key in losses)
            assert torch.isfinite(loss)
            loss.backward()
        timings[name] = time.monotonic() - t
        saved_bytes[name] = saved[0]
        flat_outputs = {'logits': outputs['pred_logits'], 'boxes': outputs['pred_boxes']}
        for i, auxiliary in enumerate(outputs['aux_outputs']):
            for key, value in auxiliary.items():
                flat_outputs[f'aux_{i}_{key}'] = value
        if name == 'native':
            baseline_outputs = {k: v.detach().clone() for k, v in flat_outputs.items()}
            baseline_losses = {k: v.detach().clone() for k, v in losses.items()}
            baseline_grads = {k: p.grad.clone() for k, p in model.named_parameters() if p.grad is not None}
            assert len(baseline_grads) == 345
        else:
            assert flat_outputs.keys() == baseline_outputs.keys()
            for key, value in flat_outputs.items():
                torch.testing.assert_close(value, baseline_outputs[key], rtol=0, atol=0)
                max_output_error = max(max_output_error, float((value - baseline_outputs[key]).abs().max()))
            for key, value in losses.items():
                torch.testing.assert_close(value, baseline_losses[key], rtol=0, atol=0)
                max_loss_error = max(max_loss_error, float((value - baseline_losses[key]).abs().max()))
            grads = {k: p.grad for k, p in model.named_parameters() if p.grad is not None}
            assert grads.keys() == baseline_grads.keys()
            for key, grad in grads.items():
                assert torch.isfinite(grad).all(), key
                torch.testing.assert_close(grad, baseline_grads[key], rtol=1e-5, atol=1e-6, msg=key)
                max_gradient_error = max(max_gradient_error, float((grad - baseline_grads[key]).abs().max()))
        del outputs, losses, loss, flat_outputs
        gc.collect()
    assert saved_bytes['checkpointed'] < saved_bytes['native']
    assert all('forward' not in layer.__dict__ for group in
               [model.transformer.encoder.layers, model.transformer.decoder.layers] for layer in group)
    assert not getattr(model, '_fresh_checkpoint_context', False)
    try:
        with fresh_transformer_checkpointing(model, root):
            try:
                with fresh_transformer_checkpointing(model, root):
                    raise AssertionError('nested_guard_failed')
            except RuntimeError as error:
                assert str(error) == 'nested_checkpoint_context'
            raise ValueError('fixture_restoration')
    except ValueError as error:
        assert str(error) == 'fixture_restoration'
    assert all('forward' not in layer.__dict__ for group in
               [model.transformer.encoder.layers, model.transformer.decoder.layers] for layer in group)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    assert peak < 4 * 1024 ** 3
    report.update(status='PASS', compared_gradient_tensors=len(baseline_grads),
                  max_output_absolute_error=max_output_error, max_loss_absolute_error=max_loss_error,
                  max_gradient_absolute_error=max_gradient_error,
                  forward_saved_tensor_bytes_including_repeated_parameter_saves=saved_bytes,
                  forward_backward_seconds=timings, peak_cpu_rss_bytes=peak,
                  train_mode_dropout_rng_preserved=True, nested_and_exception_restoration='PASS',
                  torch_version=torch.__version__,
                  remaining=['Native-resolution GPU probe in reserved/process-memory-admitted window',
                             'GPU gradient parity and full optimizer/resource throughput',
                             'Final clean pixel release and separate fresh-training preregistration'])
except Exception as error:
    import traceback
    (folder / 'failure.log').write_text(traceback.format_exc())
    report['error'] = str(error)
    raise
finally:
    report.update(utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic()-started)
    raw = json.dumps(report, indent=2)
    (folder / 'outcome.json').write_text(raw)
    (root / 'reports/FRESH_CHECKPOINT_CPU_PROBE.json').write_text(raw)
    print(json.dumps(report))
