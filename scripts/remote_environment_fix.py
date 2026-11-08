"""Finalize the project overlay configuration without changing shared Python packages."""
import venv,subprocess,os,json
from pathlib import Path
root=Path.home()/'rsg-former'
venv.EnvBuilder(system_site_packages=True,with_pip=False).create(root/'.venv')
env=dict(os.environ,PYTHONPATH=str(root/'.tools/deps')+':'+str(root/'src'),PYTHONDONTWRITEBYTECODE='1')
code="import torch,torchvision,numpy,h5py,sys,json;assert torch.__version__=='2.12.0+cu130';assert torchvision.__version__=='0.27.0+cu130';print(json.dumps({'python':sys.version.split()[0],'executable':sys.executable,'torch':torch.__version__,'torchvision':torchvision.__version__,'numpy':numpy.__version__,'h5py':h5py.__version__,'cuda_available':torch.cuda.is_available(),'venv_isolation':'project overlay inherits locked shared torch; own h5py target'}))"
result=subprocess.run([str(root/'.venv/bin/python'),'-B','-c',code],env=env,capture_output=True,text=True)
if result.returncode:
    print(json.dumps({'status':'error','error':'environment_version_or_import_check_failed'}));raise SystemExit(1)
report=json.loads(result.stdout);(root/'reports/P0_ENVIRONMENT_LOCK.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
