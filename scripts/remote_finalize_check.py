"""Read-only final source/data/environment/checkpoint integrity in the training env."""
from pathlib import Path
import json,hashlib,sys,os
ROOT=Path.home()/'rsg-former'
sys.path[:0]=[str(ROOT/'.tools/deps'),str(ROOT/'src')]
import torch,torchvision,numpy,h5py
assert (torch.__version__,torchvision.__version__,numpy.__version__,h5py.__version__)==('2.12.0+cu130','0.27.0+cu130','2.2.6','3.15.1')
hashfile=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
pilot=json.loads((ROOT/'reports/P2_PILOT_REPORT.json').read_text())
tests=json.loads((ROOT/'reports/P2_TEST_REPORT.json').read_text())
prep=json.loads((ROOT/'reports/P0_PREPARATION.json').read_text())
prereg=json.loads((ROOT/'reports/P2_PILOT_PREREGISTRATION.json').read_text())
assert pilot['config_sha256']==prereg['config_sha256']==hashfile(ROOT/'configs/p2_pilot.json')
for f in prep['files']:
    parent='star' if f['file'].startswith('STAR') else 'vg150'
    assert hashfile(ROOT/'data'/parent/f['file'])==f['sha256']
for name in ['A0','A1']:
    run=pilot['runs'][name];path=ROOT/run['checkpoint']
    assert hashfile(path)==run['checkpoint_sha256']
    payload=torch.load(path,weights_only=True,map_location='cpu')
    assert payload['config']==pilot['config'] and payload['split_hashes']==pilot['split_hashes'] and payload['best_epoch']==run['selected_epoch']
assert tests['status']=='PASS' and tests['official_metric_parity_cases']==32
files=[]
for folder in ['src','tests','configs','scripts']:
    for p in (ROOT/folder).rglob('*'):
        if p.is_file() and p.suffix in ['.py','.json']:files.append({'file':p.relative_to(ROOT).as_posix(),'sha256':hashfile(p)})
out={'status':'PASS','interpreter':str(Path(sys.executable)),'download_checksums_verified':4,'checkpoint_checksums_verified':2,'reference_parity_cases':32,'config_matches_preregistration':True,'source_files':files,'image_ZIPs_still_present':all((Path.home()/'sbir/hienlth/vg'/p).is_file() for p in ['images.zip','images2.zip']),'current_artifact_bytes':sum(p.stat().st_size for p in (ROOT/'artifacts').rglob('*') if p.is_file()),'project_directory_write_access':os.access(ROOT,os.W_OK)}
(ROOT/'reports/FINAL_SSH_INTEGRITY_CHECK.json').write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k!='source_files'}))
