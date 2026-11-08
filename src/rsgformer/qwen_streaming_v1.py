"""Exact BF16 safetensor streaming loader; bounded CPU residency, no quantization."""
from pathlib import Path
import gc,hashlib,json

def load_qwen(folder,allocator_gib=32,verify_weights=True):
    import torch
    from transformers import AutoConfig,AutoModelForCausalLM,AutoTokenizer,GenerationConfig
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from safetensors import safe_open
    folder=Path(folder);root=Path.home()/'rsg-former'
    runtime=json.loads((root/'configs/qwen_runtime.json').read_text())
    for name,expected in runtime['auxiliary_files_sha256'].items():
        assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==expected,name
    torch.cuda.set_per_process_memory_fraction(allocator_gib*1024**3/torch.cuda.get_device_properties(0).total_memory)
    cfg=AutoConfig.from_pretrained(folder,local_files_only=True);cfg._attn_implementation='sdpa'
    with init_empty_weights():model=AutoModelForCausalLM.from_config(cfg,dtype=torch.bfloat16)
    expected=dict(model.named_parameters());index=json.loads((folder/'model.safetensors.index.json').read_text())
    assert set(index['weight_map'])==set(expected)
    loaded=set()
    for filename in sorted(set(index['weight_map'].values())):
        assert Path(filename).name==filename
        if verify_weights:
            h=hashlib.sha256()
            with (folder/filename).open('rb') as stream:
                for b in iter(lambda:stream.read(8*1024**2),b''):h.update(b)
            assert h.hexdigest()==runtime['full_weights_sha256'][filename],filename
        with safe_open(folder/filename,framework='pt',device='cpu') as stream:
            for name in stream.keys():
                value=stream.get_tensor(name)
                assert tuple(value.shape)==tuple(expected[name].shape) and value.dtype==torch.bfloat16
                set_module_tensor_to_device(model,name,'cuda:0',value=value,dtype=torch.bfloat16)
                loaded.add(name);del value;gc.collect()
        gc.collect()
    assert loaded==set(expected) and not any(p.is_meta for p in model.parameters())
    model.to('cuda:0').eval();model.generation_config=GenerationConfig.from_pretrained(folder,local_files_only=True)
    tokenizer=AutoTokenizer.from_pretrained(folder,local_files_only=True)
    return model,tokenizer
