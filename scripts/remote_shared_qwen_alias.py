"""Legacy entrypoint: register the canonical existing Qwen directory; no shared weight alias."""
from pathlib import Path
import json
home=Path.home();shared=home/'shared-assets'
target=home/'etechs/msvd-tkg-captioning/models/qwen2.5-14b'
assert target.resolve()==target and target.is_dir()
registry=json.loads((shared/'ASSET_REGISTRY.json').read_text())
for key in ['path','load_path']:registry['qwen14b'][key]='${REMOTE_HOME}/etechs/msvd-tkg-captioning/models/qwen2.5-14b'
for p in [shared/'ASSET_REGISTRY.json',home/'rsg-former/reports/SHARED_ASSET_REGISTRY.json']:
    p.write_text(json.dumps(registry,indent=2))
print(json.dumps({'short_load_path':registry['qwen14b']['load_path'],'weight_copy_created':False,'shared_qwen_alias_created':False}))
