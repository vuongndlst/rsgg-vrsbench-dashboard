"""Locate accidental credential strings without printing values or source lines."""
from pathlib import Path
import re,shlex,json
root=Path(__file__).absolute().parents[1]
env={m.group(1):m.group(2).strip().strip('"').strip("'") for m in re.finditer(r'^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$',(root/'.env').read_text(),re.M)}
secrets={k:v for k,v in env.items() if len(v)>=6}
for part in shlex.split(env.get('SSH_SERVER','')):
    if '@' in part and not part.startswith('-'):
        account,host=part.rsplit('@',1);secrets.update(SSH_ACCOUNT=account,SSH_HOST=host)
hits=[]
for path in root.rglob('*'):
    if not path.is_file() or '.tools' in path.parts or path.suffix not in ['.json','.jsonl','.md','.py','.csv','.txt','.bib','.html']:continue
    text=path.read_text(encoding='utf-8',errors='replace')
    lines=[]
    for n,line in enumerate(text.splitlines(),1):
        keys=[k for k,v in secrets.items() if v and v in line]
        if keys:lines.append({'line':n,'matched_setting_keys':keys})
    if lines:hits.append({'file':str(path.relative_to(root)),'lines':lines[:10]})
print(json.dumps(hits))
