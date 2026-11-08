"""One registered, discarded train-image optimizer step under a 4GiB GPU cap.

Not a dataset training run or clean-weight release. Never touches live v2 code.
"""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time

root = Path.home() / 'rsg-former'
folder = root / 'artifacts/caption_readiness/fresh_native_gpu_probe_v1'
assert not folder.exists(), 'preserve_existing_probe_version_no_hidden_retry'
assert shutil.disk_usage(root).free > 16 * 1024 ** 3 + 64 * 1024 ** 2
progress = json.loads((root / 'reports/VG_V2_PROGRESS.json').read_text())
assert progress['stage'] == 'training', 'only_admit_during_stable_cached_head_training'
assert progress['run'] == 'screening_native_1729'
assert (datetime.now(timezone.utc) - datetime.fromisoformat(progress['utc'])).total_seconds() < 120
assert progress['peak_allocated_bytes'] + 4 * 1024 ** 3 < 8 * 1024 ** 3
manifest_path = root / 'artifacts/caption_readiness/canonical_internal_external_identity_splits_v2.json'
manifest_raw = manifest_path.read_bytes()
assert hashlib.sha256(manifest_raw).hexdigest() == 'ade090fd4975e12e1ade2ac16a734f7b167f6630b38d1710cefa4176523f7b16'
identity = min(json.loads(manifest_raw)['split_image_ids']['train'])
cpu = json.loads((root / 'reports/FRESH_DETECTOR_CPU_PROBE.json').read_text())
assert cpu['status'] == 'PASS'
hashes = dict(cpu['source_hashes'])
hashes.update({
    'src/rsgformer/vg_canonical_data.py': 'f14af9890c36d59531466372f05c6ccacc38879e10711ddacff61857be1914e6',
    'src/rsgformer/vg_canonical_training.py': '5474d999d9a4306b263e69efa0919b5af5996565dbb282532ac30ec93993ee84',
    'vendor/salience_sgg/datasets/coco.py': '429ba75358fa90c89b70a7ca2a339e2991b9206fba06cb88fd3f042d0eee88a1',
    'vendor/salience_sgg/datasets/transforms.py': 'b96788059ed7146be09f8724434cb06d3ac49c816613a5fafb019d557c751a9d',
})
for name, expected in hashes.items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
hashes['scripts/remote_fresh_detector_native_gpu_probe.py'] = hashlib.sha256((root / 'scripts/remote_fresh_detector_native_gpu_probe.py').read_bytes()).hexdigest()
protocol = {
    'schema_version': 1, 'utc': datetime.now(timezone.utc).isoformat(),
    'experiment_role': 'train-only native-resolution compatibility/resource smoke, not clean dataset training',
    'image_selection': 'minimum retained TRAIN ID in pinned provisional draft; no score-based selection',
    'vg_train_image_id': identity, 'provisional_manifest_sha256': hashlib.sha256(manifest_raw).hexdigest(),
    'clean_release_certified': False, 'source_hashes': hashes,
    'seed': 1729, 'precision': 'FP32',
    'spatial_recipe': 'pinned native detection validation resize800/max1333 + ImageNet normalization on TRAIN image; no random crop',
    'batch_size': 1, 'optimizer_steps': 1,
    'optimizer': {'name': 'AdamW', 'lr': 1e-4, 'backbone_lr': 1e-5, 'weight_decay': 1e-4, 'gradient_clip': .1},
    'resource': {'cuda_allocator_gib': 4, 'aggregate_own_allocator_gib': 8,
                 'cpu_threads': 1, 'timeout_seconds': 120, 'minimum_free_disk_gib': 16,
                 'minimum_free_gpu_bytes_before_load': 5 * 1024 ** 3},
    'admission_live_run': progress['run'], 'admission_live_peak_allocated_bytes': progress['peak_allocated_bytes'],
    'vg_author_checkpoint_loaded': False, 'checkpoint_or_optimizer_state_saved': False,
    'model_weights_discarded_after_smoke': True, 'benchmark_test_accessed': False,
    'scientific_training_recipe_registered': False,
}
folder.mkdir()
(folder / 'preregistration.json').write_text(json.dumps(protocol, indent=2))
child = r'''
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, os, resource, sys, time
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
root, folder = Path(sys.argv[1]), Path(sys.argv[2])
protocol = json.loads((folder / 'preregistration.json').read_text())
sys.path.insert(0, str(root / 'src'))
os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.set_float32_matmul_precision('highest')
started = time.monotonic()
device = torch.device('cuda:0')
properties = torch.cuda.get_device_properties(device)
torch.cuda.set_per_process_memory_fraction(4*1024**3/properties.total_memory, device)
free, total = torch.cuda.mem_get_info(device)
assert free >= protocol['resource']['minimum_free_gpu_bytes_before_load'], 'shared_gpu_free_memory_admission_failed'
from rsgformer.vg_fresh_detector import build_fresh_detector
from rsgformer.vg_canonical_data import VGCanonical
from rsgformer.vg_canonical_training import native_components
model, criterion, post, config, provenance = build_fresh_detector(root, seed=protocol['seed'])
ds = VGCanonical(root / 'data/vg_canonical_audit/reltr_vg_annotations.zip',
                 Path.home() / 'sbir/hienlth/vg', split='train')
identity = protocol['vg_train_image_id']
index = ds.ids.index(identity)
sample = ds[index]
assert sample.image_id == identity
prepare, transform = native_components(root, 'detection', 'val')
image, target = prepare(ds.image(index), {'image_id': identity, 'annotations': ds.annotations[identity]})
image, target = transform(image, target)
target['labels'] -= 1
assert min(image.shape[-2:]) == 800 or max(image.shape[-2:]) in (1332,1333)
assert max(image.shape[-2:]) <= 1333
ds.close()
from util.misc import nested_tensor_from_tensor_list
model, criterion = model.to(device).train(), criterion.to(device)
samples = nested_tensor_from_tensor_list([image.to(device)])
targets = [{key: target[key].to(device) for key in ('labels','boxes')}]
parameter_groups = [
    {'params':[parameter for name,parameter in model.named_parameters() if parameter.requires_grad and 'backbone' not in name]},
    {'params':[parameter for name,parameter in model.named_parameters() if parameter.requires_grad and 'backbone' in name], 'lr':1e-5},
]
optimizer = torch.optim.AdamW(parameter_groups, lr=1e-4, weight_decay=1e-4)
before_step = {name:parameter.detach().clone() for name,parameter in model.named_parameters()
               if name.endswith('class_embed.0.weight') or name.endswith('bbox_embed.0.layers.2.weight')}
assert before_step, 'specific_head_update_witness_missing'
torch.cuda.synchronize()
forward_started=time.monotonic()
outputs=model(samples)
losses=criterion(outputs,targets)
loss=sum(losses[key]*weight for key,weight in criterion.weight_dict.items() if key in losses)
assert torch.isfinite(loss), 'nonfinite_smoke_loss'
torch.cuda.synchronize()
forward_seconds=time.monotonic()-forward_started
backward_started=time.monotonic()
loss.backward()
groups={key:0 for key in ('backbone','transformer','class_head','box_head')}
finite_count=0
for name,parameter in model.named_parameters():
    if parameter.grad is None:continue
    assert torch.isfinite(parameter.grad).all(), 'nonfinite_smoke_gradient'
    finite_count+=1
    group=next((key for key,token in [('class_head','class_embed'),('box_head','bbox_embed'),('backbone','backbone'),('transformer','transformer')] if token in name),None)
    if group:groups[group]+=int(bool(torch.count_nonzero(parameter.grad)))
assert all(value>0 for value in groups.values()), 'missing_smoke_gradient_group'
torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], .1)
optimizer.step()
torch.cuda.synchronize()
backward_optimizer_seconds=time.monotonic()-backward_started
updated=sum(int(not torch.equal(before,dict(model.named_parameters())[name])) for name,before in before_step.items())
assert updated>0, 'optimizer_did_not_update_native_head'
peak=torch.cuda.max_memory_allocated(device)
live=json.loads((root / 'reports/VG_V2_PROGRESS.json').read_text())
assert live['stage']=='training' and live['run']==protocol['admission_live_run'], 'live_stage_changed_during_probe'
assert peak<=4*1024**3 and peak+live['peak_allocated_bytes']<8*1024**3, 'aggregate_allocator_gate'
report=dict(provenance,status='PASS',vg_train_image_id=identity,image_shape=list(image.shape),
            forward_loss_seconds=forward_seconds,backward_optimizer_seconds=backward_optimizer_seconds,
            elapsed_seconds=time.monotonic()-started,peak_cuda_allocated_bytes=peak,
            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(device),
            live_v2_peak_cuda_allocated_bytes=live['peak_allocated_bytes'],
            summed_own_observed_peak_allocated_bytes=peak+live['peak_allocated_bytes'],
            allocator_cap_gib=4,aggregate_allocator_cap_gib=8,
            peak_cpu_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            finite_gradient_tensors=finite_count,nonzero_gradient_groups=groups,
            native_head_witness_tensors_updated=updated,optimizer_step_executed=True,
            microbatch=1,precision='FP32',actual_train_image_used=True,
            dataset_training_launched=False,checkpoint_saved=False,clean_release_certified=False,
            benchmark_validation_scored=False,benchmark_test_scored=False,live_v2_modified=False,
            resource_gate_scope='one TRAIN image, fixed800/1333; no full-dataset throughput/memory bound')
print(json.dumps(report))
'''
env = dict(os.environ)
env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
started = time.monotonic()
process = subprocess.Popen([sys.executable,'-B','-c',child,str(root),str(folder)],
                           env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                           text=True,start_new_session=True)
timed_out=False
try:
    output,diagnostic=process.communicate(timeout=120)
except subprocess.TimeoutExpired:
    timed_out=True
    os.killpg(process.pid,signal.SIGTERM)
    try:output,diagnostic=process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid,signal.SIGKILL)
        output,diagnostic=process.communicate(timeout=5)
(folder/'stdout.log').write_text(output)
(folder/'stderr.log').write_text(diagnostic)
if process.returncode==0:
    report=json.loads(output.splitlines()[-1])
else:
    report={'status':'RESOURCE_OR_COMPATIBILITY_PROBE_FAILED','exit_code':process.returncode,
            'timed_out':timed_out,'diagnostic_tail':diagnostic[-3000:],
            'cuda_oom':'OutOfMemoryError' in diagnostic or 'CUDA out of memory' in diagnostic,
            'dataset_training_launched':False,'checkpoint_saved':False,
            'benchmark_test_scored':False,'clean_release_certified':False,'live_v2_modified':False}
report.update(utc=datetime.now(timezone.utc).isoformat(),parent_elapsed_seconds=time.monotonic()-started,
              protocol=protocol,preregistration_sha256=hashlib.sha256((folder/'preregistration.json').read_bytes()).hexdigest(),
              evidence_folder_alias='${REMOTE_HOME}/rsg-former/artifacts/caption_readiness/fresh_native_gpu_probe_v1',
              remaining=['Full protected pixel/exclusion release before fresh training',
                         'Representative train-only image shapes and measured full training throughput/resource budget',
                         'Separate fresh detector/SGG train/validation recipe and selection preregistration'])
(root/'reports/FRESH_DETECTOR_NATIVE_GPU_PROBE.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report))
if process.returncode!=0:sys.exit(1)
