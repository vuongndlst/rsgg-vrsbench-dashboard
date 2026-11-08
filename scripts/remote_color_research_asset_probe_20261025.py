"""Read-only lookup of existing SAM2 assets and dependencies; no model load."""
from pathlib import Path
import hashlib,importlib.util,json
home=Path.home();directory=home/'etechs/thesis/var/models/sam2';out={'sam2_directory_exists':directory.exists(),'files':[],'modules':{}}
if directory.exists():
 for p in directory.rglob('*'):
  if p.is_file():
   row={'path':str(p),'bytes':p.stat().st_size}
   if p.suffix in ['.pt','.pth','.yaml','.json'] and p.stat().st_size<300*1024**2:row['sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
   out['files'].append(row)
for module in ['sam2','hydra','omegaconf','scipy','cv2','torch']:
 try:spec=importlib.util.find_spec(module);out['modules'][module]=spec.origin if spec else None
 except Exception as e:out['modules'][module]={'error':type(e).__name__}
project=home/'etechs/thesis'
out['likely_source_directories']=[str(p) for pattern in ['*sam*','third_party/*sam*','vendor/*sam*'] for p in project.glob(pattern) if p.is_dir()]
print(json.dumps(out))
