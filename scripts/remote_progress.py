import json,time
from pathlib import Path
root=Path.home()/'rsg-former'
out={'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'weights':[],'feature_cache_files':0,'feature_cache_bytes':0}
for p in (root/'weights').rglob('*'):
    if p.is_file():out['weights'].append({'name':p.name,'bytes':p.stat().st_size})
for p in (root/'artifacts').glob('features_*.pt'):
    out['feature_cache_files']+=1;out['feature_cache_bytes']+=p.stat().st_size
if (root/'reports/P2_PILOT_REPORT.json').exists():
    d=json.loads((root/'reports/P2_PILOT_REPORT.json').read_text());out['baselines_finished']=list(d.get('runs',{}));out['status']=d.get('status','running')
print(json.dumps(out))
