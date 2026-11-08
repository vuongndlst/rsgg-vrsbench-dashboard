"""Read resources and prepare an isolated, pinned upstream source checkout."""
from pathlib import Path
import subprocess,json,os,pwd,shutil,datetime
root=Path.home()/'rsg-former'; vendor=root/'vendor'/'salience_sgg'
def run(argv,timeout=60):
 p=subprocess.run(argv,capture_output=True,text=True,timeout=timeout)
 return {'returncode':p.returncode,'stdout':p.stdout[-18000:],'stderr':p.stderr[-2000:]}
def safe(s):
 return s.replace(str(Path.home()),'${REMOTE_HOME}').replace(pwd.getpwuid(os.getuid()).pw_name,'${ACCOUNT}')
out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'disk':dict(zip(['total','used','free'],shutil.disk_usage(root))),
 'gpu':run(['nvidia-smi','--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader']),
 'gpu_processes':run(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader']),
 'system':run(['bash','-c','command -v nvcc; command -v uv; command -v conda; command -v python3; free -m']),
 'vg_files':[{'name':p.name,'bytes':p.stat().st_size} for p in (Path.home()/'sbir/hienlth/vg').iterdir() if p.is_file()]}
vendor.parent.mkdir(exist_ok=True)
if not vendor.exists():out['clone']=run(['git','clone','--depth','1','https://github.com/runfeng-q/Salience-SGG.git',str(vendor)],120)
out['commit']=run(['git','-C',str(vendor),'rev-parse','HEAD'])
out['files']=run(['git','-C',str(vendor),'ls-files'])
for name in ['ReadME.md','configs/SalienceSGG_vg.yaml','configs/DABDeformableDETR_vg.yaml','run.py']:
 p=vendor/name
 if p.exists():out[name]=p.read_text()
out['project_environment']=run([str(root/'.venv/bin/python'),'-c',"import torch,torchvision,sys;print(sys.version);print(torch.__version__,torchvision.__version__);print(torch.cuda.is_available());import importlib.util;print({x: bool(importlib.util.find_spec(x)) for x in ['transformers','lightning','pycocotools','gdown','timm','yaml','cv2','scipy']})"])
text=safe(json.dumps(out,indent=2));(root/'reports/VG_V2_PREFLIGHT.json').write_text(text);print(text)
