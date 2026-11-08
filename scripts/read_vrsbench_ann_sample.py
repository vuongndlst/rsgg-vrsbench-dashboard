import zipfile, json
from pathlib import Path

target = Path.home() / 'shared-assets/datasets/vrsbench/Annotations_train.zip'
with zipfile.ZipFile(target) as z:
    for name in z.namelist():
        if name.endswith('.json') and '__MACOSX' not in name:
            data = json.loads(z.read(name).decode('utf-8'))
            print(f"File: {name}")
            print(json.dumps(data, indent=2))
            break

