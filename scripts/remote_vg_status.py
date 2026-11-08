"""Compact VG experiment status; no credentials, other processes, or test scoring."""
from pathlib import Path
import json,os
ROOT=Path.home()/'rsg-former';folder=ROOT/'artifacts/vg_method_v1';out={}
pidfile=folder/'pipeline.pid'
if pidfile.exists():
    pid=int(pidfile.read_text());p=Path(f'/proc/{pid}/cmdline');out['pid']=pid;out['process_active']=p.exists() and b'remote_vg_method.py' in p.read_bytes()
for name in ['VG_METHOD_PROGRESS.json','VG_METHOD_RESULT.json','VG_FULL_FREQUENCY.json']:
    path=ROOT/'reports'/name
    if path.exists():
        v=json.loads(path.read_text());out[name]=v if name!='VG_FULL_FREQUENCY.json' else {'R50':v['gc']['R@50'],'mR50':v['gc']['mR@50']}
manifest=folder/'features/manifest.json'
if manifest.exists():
    m=json.loads(manifest.read_text());out['cache']={'status':m['status'],'images':{s:sum(len(e['image_ids']) for e in entries) for s,entries in m['splits'].items()},'bytes':sum(e['bytes'] for entries in m['splits'].values() for e in entries)}
for phase in ['screening','confirmation']:
    reports=[]
    for path in sorted(folder.glob(phase+'*/report.json')):
        r=json.loads(path.read_text());v={'run':r['run_id'],'status':r['status'],'epochs_finished':len(r['history']),'selected_epoch':r['selected_epoch'],'report_updated_unix':path.stat().st_mtime,'epoch_seconds':[h['seconds'] for h in r['history']]}
        if r['history']:v['last_epoch']={k:r['history'][-1][k] for k in ['epoch','gc_R50','gc_mR50','train_loss','seconds']}
        if r['status']=='complete':
            v['checkpoint']=r['checkpoint'];v['checkpoint_sha256']=r['checkpoint_sha256'];v['wall_seconds']=r['wall_seconds']
            v['selected_metrics']={k:r['selected_metrics']['gc'][k] for k in ['R@50','mR@50']}
        reports.append(v)
    if reports:out[phase]=reports
log=folder/'pipeline.log'
if log.exists():
    with log.open('rb') as f:f.seek(max(0,log.stat().st_size-16000));lines=f.read().decode(errors='replace').splitlines()
    out['log_tail']=lines[-10:]
(ROOT/'reports/VG_METHOD_STATUS.json').write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)
