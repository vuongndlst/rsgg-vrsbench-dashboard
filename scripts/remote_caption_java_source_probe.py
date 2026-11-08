"""Read installed scorer implementation before bounded Java fixture execution."""
import importlib.util
from pathlib import Path
import json
modules = ['pycocoevalcap.meteor.meteor', 'pycocoevalcap.spice.spice',
           'pycocoevalcap.tokenizer.ptbtokenizer', 'pycocoevalcap.spice.get_stanford_models']
result = {}
for module in modules:
    source = Path(importlib.util.find_spec(module).origin)
    text = source.read_text()
    assert len(text) < 25000
    result[module] = text
print(json.dumps(result))
