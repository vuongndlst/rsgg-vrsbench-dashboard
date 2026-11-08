"""Offline header-only Qwen memory audit and tiny CPU NF4 compatibility probe."""
import os
for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[name] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
from pathlib import Path
from datetime import datetime, timezone
import contextlib
import hashlib
import importlib.metadata
import inspect
import io
import json
import math
import signal
import struct

signal.alarm(90)
os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
root = Path.home() / 'rsg-former'
model = Path.home() / 'etechs/msvd-tkg-captioning/models/qwen2.5-14b'
index_raw = (model / 'model.safetensors.index.json').read_bytes()
assert hashlib.sha256(index_raw).hexdigest() == '46627f7a9e851d9cede6a7a4c49999082148ced9c22d2b9cfaaa5ba8b65dc68f'
index = json.loads(index_raw)
config_raw = (model / 'config.json').read_bytes()
config = json.loads(config_raw)
assert config['model_type'] == 'qwen2' and not config['tie_word_embeddings']
headers, tensors = [], {}
for filename in sorted(set(index['weight_map'].values())):
    assert Path(filename).name == filename and filename.endswith('.safetensors')
    path = model / filename
    with path.open('rb') as stream:
        length_bytes = stream.read(8)
        length = struct.unpack('<Q', length_bytes)[0]
        assert 0 < length <= 2 * 1024 ** 2
        raw = stream.read(length)
    assert len(raw) == length
    header = json.loads(raw)
    headers.append({'filename': filename, 'bytes': path.stat().st_size,
                    'header_bytes': length + 8, 'header_sha256': hashlib.sha256(length_bytes + raw).hexdigest()})
    for name, value in header.items():
        if name == '__metadata__':
            continue
        assert name not in tensors and index['weight_map'][name] == filename
        assert value['dtype'] == 'BF16'
        count = math.prod(value['shape'])
        start, end = value['data_offsets']
        assert end - start == count * 2 and 0 <= start < end <= path.stat().st_size - 8 - length
        tensors[name] = {'shape': value['shape'], 'elements': count}
assert set(tensors) == set(index['weight_map']) and len(headers) == 8
groups = {'embedding': 0, 'lm_head': 0, 'linear_matrices': 0, 'norm_bias_other': 0}
quant_count = 0
nf4_payload = nf4_double_scale = 0
for name, value in tensors.items():
    count = value['elements']
    group = 'embedding' if name == 'model.embed_tokens.weight' else 'lm_head' if name == 'lm_head.weight' else 'linear_matrices' if len(value['shape']) == 2 else 'norm_bias_other'
    groups[group] += count
    if group == 'linear_matrices':
        quant_count += 1
        nf4_payload += (count + 1) // 2
        blocks = (count + 63) // 64
        # Payload + one uint8 first-level absmax/block + float32 second-level scales.
        # Excludes codebooks, offsets, alignments, kernels and allocator overhead.
        nf4_double_scale += blocks + 4 * ((blocks + 255) // 256)
elements = sum(groups.values())
assert elements * 2 == index['metadata']['total_size']
hidden, layers = config['hidden_size'], config['num_hidden_layers']
kv_heads, head_dim = config['num_key_value_heads'], hidden // config['num_attention_heads']
kv = {str(tokens): 2 * 2 * layers * kv_heads * head_dim * tokens for tokens in [512, 1024, 2048, 4096]}
cap = 8 * 1024 ** 3
base_nf4 = nf4_payload + nf4_double_scale
default_gpu_lower_bound = base_nf4 + 2 * (groups['embedding'] + groups['lm_head'] + groups['norm_bias_other'])
offload_head_lower_bound = base_nf4 + 2 * (groups['embedding'] + groups['norm_bias_other'])
offload_both_lower_bound = base_nf4 + 2 * groups['norm_bias_other']

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
from transformers import BitsAndBytesConfig
from transformers.quantizers.quantizer_bnb_4bit import Bnb4BitHfQuantizer
quantizer_source = Path(inspect.getfile(Bnb4BitHfQuantizer))
source_text = quantizer_source.read_text()
quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                                 bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
                                 llm_int8_enable_fp32_cpu_offload=True)
cpu_probe = {}
try:
    with contextlib.redirect_stdout(io.StringIO()):
        import bitsandbytes as bnb
        toy = torch.linspace(-1, 1, 4096, dtype=torch.float32).reshape(64, 64)
        packed, state = bnb.functional.quantize_4bit(toy, blocksize=64, compress_statistics=True, quant_type='nf4')
        restored = bnb.functional.dequantize_4bit(packed, quant_state=state)
    assert packed.device.type == restored.device.type == 'cpu'
    assert packed.dtype == torch.uint8 and packed.numel() == 2048
    assert restored.shape == toy.shape and torch.isfinite(restored).all()
    cpu_probe = {'status': 'PASS', 'scope': '4096 invented CPU values; no model weights or GPU',
                 'packed_bytes': packed.numel() * packed.element_size(),
                 'maximum_absolute_reconstruction_error': float((restored - toy).abs().max()),
                 'model_accuracy_verified': False}
except Exception as error:
    cpu_probe = {'status': 'NOT_AVAILABLE', 'error_type': type(error).__name__, 'error': str(error),
                 'scope': 'tiny CPU compatibility only; not GPU inference failure'}

report = {'status': 'HEADER_ONLY_QUANTIZATION_PREFLIGHT_COMPLETE', 'utc': datetime.now(timezone.utc).isoformat(),
          'versions': {name: importlib.metadata.version(name) for name in ['torch', 'transformers', 'accelerate', 'bitsandbytes']},
          'config_sha256': hashlib.sha256(config_raw).hexdigest(), 'index_sha256': hashlib.sha256(index_raw).hexdigest(),
          'safetensors_headers': headers, 'parameter_elements': elements, 'parameter_groups': groups,
          'linear_matrices_considered_for_nf4': quant_count,
          'bf16_tensor_payload_bytes': elements * 2, 'nf4_linear_payload_plus_double_scale_lower_bound_bytes': base_nf4,
          'all_gpu_nf4_except_embedding_head_bf16_lower_bound_bytes': default_gpu_lower_bound,
          'all_gpu_lower_bound_exceeds_8GiB_before_KV_overhead': default_gpu_lower_bound > cap,
          'head_cpu_embedding_gpu_lower_bound_bytes': offload_head_lower_bound,
          'head_and_embedding_cpu_gpu_lower_bound_bytes': offload_both_lower_bound,
          'bf16_embedding_and_head_cpu_payload_bytes': 2 * (groups['embedding'] + groups['lm_head']),
          'fp32_embedding_and_head_cpu_payload_bytes': 4 * (groups['embedding'] + groups['lm_head']),
          'bf16_KV_cache_bytes_batch1_by_total_tokens': kv,
          'estimate_excludes': ['allocator, activation and workspace memory', 'quantization transient allocations',
                                'scale codebooks/alignment', 'CPU dispatch dtype/runtime behavior', 'other live jobs'],
          'quantizer_source_sha256': hashlib.sha256(quantizer_source.read_bytes()).hexdigest(),
          'quantizer_source_mentions_cpu_offload_guard': 'llm_int8_enable_fp32_cpu_offload' in source_text,
          'quantization_config_constructed_offline': quant_config.to_dict(),
          'tiny_cpu_nf4_probe': cpu_probe, 'cuda_used': False, 'model_weights_loaded': False,
          'model_parameters_instantiated': False, 'generation_executed': False,
          'dataset_references_read': False, 'benchmark_test_scored': False,
          'shared_assets_or_environment_modified': False, 'live_v2_modified': False,
          'resource_gate_passed': False,
          'remaining': ['Explicit offload map and measured native model load/generation pilot within aggregate8GiB',
                        'Do not co-load 14B with live v2 merely because theoretical storage fits',
                        'Lock quantization/CPU dtype/decoding/token budgets consistently across caption arms',
                        'Measured throughput and memory, not estimates, govern full evaluation']}
(root / 'reports/QWEN_QUANTIZATION_PREFLIGHT.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report))
