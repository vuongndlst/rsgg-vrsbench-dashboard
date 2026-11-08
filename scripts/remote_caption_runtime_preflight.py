"""Offline caption runtime preparation; synthetic strings, no model weights or images."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import contextlib
import importlib.metadata
import importlib.util
import inspect
import io
import json
import shutil
import signal
import subprocess
import time

signal.alarm(120)
started = time.monotonic()
root = Path.home() / 'rsg-former'
result = {
    'utc': datetime.now(timezone.utc).isoformat(),
    'scope': 'offline CPU synthetic scorer/tokenizer preparation only',
    'cuda_used': False, 'model_weights_loaded': False,
    'dataset_references_read': False, 'benchmark_validation_scored': False,
    'benchmark_test_scored': False, 'network_used': False,
    'live_v2_modified': False, 'versions': {}, 'scorers': {},
}
for name in ['transformers', 'accelerate', 'bitsandbytes', 'pycocoevalcap', 'pycocotools']:
    try:
        result['versions'][name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        result['versions'][name] = None

java = shutil.which('java')
if java:
    completed = subprocess.run([java, '-version'], capture_output=True, text=True, timeout=10)
    result['java'] = {'available': completed.returncode == 0,
                      'version_first_line': (completed.stderr or completed.stdout).splitlines()[0]}
else:
    result['java'] = {'available': False}

modules = {'BLEU': 'pycocoevalcap.bleu.bleu', 'CIDEr': 'pycocoevalcap.cider.cider',
           'ROUGE_L': 'pycocoevalcap.rouge.rouge', 'METEOR': 'pycocoevalcap.meteor.meteor',
           'SPICE': 'pycocoevalcap.spice.spice', 'PTBTokenizer': 'pycocoevalcap.tokenizer.ptbtokenizer'}
for name, module in modules.items():
    try:
        spec = importlib.util.find_spec(module)
        source = Path(spec.origin)
        folder = source.parent
        jars = []
        for jar in sorted(folder.rglob('*.jar')):
            jars.append({'relative_path': jar.relative_to(folder).as_posix(),
                         'bytes': jar.stat().st_size,
                         'sha256': hashlib.sha256(jar.read_bytes()).hexdigest()})
        result['scorers'][name] = {'source_available': True,
                                  'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                                  'jars': jars}
        if name in ['SPICE', 'METEOR', 'PTBTokenizer']:
            # Inspect shipped source, do not invoke Java or SPICE auto-download here.
            text = source.read_text()
            result['scorers'][name]['source_mentions_auto_download'] = 'get_stanford_models' in text
            result['scorers'][name]['runtime_executed'] = False
    except (ModuleNotFoundError, AttributeError, OSError) as error:
        result['scorers'][name] = {'source_available': False, 'error_type': type(error).__name__}

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge
# Already-tokenized invented fixtures, never drawn from a benchmark/reference file.
references = {1: ['a red ball rests on a table', 'a red ball is on a table'],
              2: ['two blue birds fly above a tree', 'two birds are flying above a tree']}
predictions = {1: [references[1][0]], 2: [references[2][0]]}
import numpy as np
for name, scorer in [('BLEU', Bleu(4)), ('CIDEr', Cider()), ('ROUGE_L', Rouge())]:
    with contextlib.redirect_stdout(io.StringIO()):
        score, per_image = scorer.compute_score(references, predictions)
    assert np.isfinite(np.asarray(score)).all()
    assert np.isfinite(np.asarray(per_image)).all()
    assert np.asarray(per_image).shape[-1] == 2
    if name == 'ROUGE_L':
        assert float(score) == 1.0
    result['scorers'][name].update(runtime_executed=True, synthetic_finite_contract='PASS')

model_path = Path.home() / 'etechs/msvd-tkg-captioning/models/qwen2.5-14b'
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
messages = [{'role': 'system', 'content': 'Describe only the supplied scene facts.'},
            {'role': 'user', 'content': 'Objects: ball, table. Relation: ball on table.'}]
tokens = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                       return_dict=False)
assert len(tokens) > 0 and all(isinstance(value, int) for value in tokens)
result['qwen_tokenizer'] = {
    'status': 'PASS', 'local_files_only': True, 'synthetic_chat_template_tokens': len(tokens),
    'tokenizer_class': type(tokenizer).__name__, 'model_weights_loaded': False,
    'chat_template_signature': str(inspect.signature(tokenizer.apply_chat_template)),
    'chat_template_return_dict_explicit': False,
    'source_hashes': {name: hashlib.sha256((model_path / name).read_bytes()).hexdigest()
                      for name in ['tokenizer.json', 'tokenizer_config.json']},
}
result['status'] = 'PARTIAL_RUNTIME_PREFLIGHT'
result['remaining'] = [
    'Execute bounded PTBTokenizer/METEOR/SPICE fixtures after Java/assets/memory gate; do not let SPICE download implicitly',
    'Frozen Qwen14B inference quantization/offload and aggregate resource pilot before model launch',
    'Preregister corpus tokenization, metric versions/options, prompt/decoding and paired image bootstrap',
    'No benchmark caption evaluation before fresh detector/graph release and selection locks',
]
result['elapsed_seconds'] = time.monotonic() - started
report = root / 'reports/CAPTION_RUNTIME_PREFLIGHT.json'
report.write_text(json.dumps(result, indent=2))
print(json.dumps(result))
