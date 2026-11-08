"""Read-only source/resource admission for a separate training-speed benchmark."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, subprocess
root = Path.home()/'rsg-former'
lock=json.loads((root/'reports/VG_V2_PREREGISTRATION.json').read_text())
assert all(hashlib.sha256((root/p).read_bytes()).hexdigest()==h for p,h in lock['source_hashes'].items())
out={'utc':datetime.now(timezone.utc).isoformat(),'source_lock_verified':True,
     'live_modified':False,'gpu_processes':subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15).stdout,
     'gpu_free':subprocess.run(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15).stdout,
     'progress':json.loads((root/'reports/VG_V2_PROGRESS.json').read_text()),'sources':{}}
for rel in ['models/salience_sgg/deformable_detr.py','models/salience_sgg/deformable_transformer.py']:
 p=root/'vendor/salience_sgg'/rel
 lines=p.read_text().splitlines()
 tokens=['def loss_relations','def loss_confidence','def forward(self, outputs, targets)','num_boxes =','if lid==(self.num_layers-1):','add_add_add_relation =','add_add_add_relation=']
 sections=[]
 for i,line in enumerate(lines):
  if any(token in line for token in tokens):sections.append({'line':i+1,'text':'\n'.join(lines[max(0,i-2):i+95])})
 out['sources'][rel]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'sections':sections}
print(json.dumps(out))
