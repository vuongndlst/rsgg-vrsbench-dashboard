"""Bounded common runtime for immutable color-only diagnostic phases."""
from pathlib import Path
from datetime import datetime,timezone
import gc,gzip,hashlib,io,json,os,resource,shutil,sqlite3,subprocess,time,zipfile
from .numpy_json_adapter_v1 import numpy_json_default

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def save(p,d):
 p=Path(p);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,indent=2,default=numpy_json_default,allow_nan=False));t.replace(p)
def line(d):return json.dumps(d,default=numpy_json_default,allow_nan=False)
def artifact_bytes(path):
 total=0;seen=set()
 for p in Path(path).rglob('*'):
  if p.is_symlink() or not p.is_file():continue
  stat=p.stat();key=(stat.st_dev,stat.st_ino)
  if key not in seen:total+=stat.st_size;seen.add(key)
 return total
class Guard:
 def __init__(self,root,art,wall,phase_gib):self.root=Path(root);self.art=Path(art);self.wall=wall;self.phase_gib=phase_gib;self.start=time.monotonic();self.storage_at=-100
 def check(self,gpu=True):
  assert time.time()<1794670321.2414355 and time.monotonic()-self.start<self.wall,'authorized_wall_limit'
  assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=4*1024**3,'RSS4GiB'
  assert shutil.disk_usage(self.root).free>=16*1024**3,'disk_reserve16GiB'
  assert not (self.art/'STOP').exists(),'owned_STOP'
  if time.monotonic()-self.storage_at>60:
   assert artifact_bytes(self.root/'artifacts')<=40*1024**3,'aggregate_artifacts40GiB'
   if self.art.exists():assert artifact_bytes(self.art)<=self.phase_gib*1024**3,'phase_artifact_cap'
   self.storage_at=time.monotonic()
  if gpu:
   q=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
   own=sum(int(x.split(',')[1])*1024**2 for x in q.splitlines() if int(x.split(',')[0])==os.getpid());assert own<=8*1024**3,'own_GPU8GiB'
def load_models(root,guard):
 root=Path(root)
 import sys
 sys.path.insert(0,str(root/'third_party/sam2_color_audit_20261025'))
 import torch,cv2
 from .vg_object_color_v2 import ClipVerifier
 from .vg_color_binding_native_v1 import NeutralBinding
 from .vg_color_fast_engine_v1 import ColorEngine,MemoColorNames
 from sam2.build_sam import build_sam2
 from sam2.sam2_image_predictor import SAM2ImagePredictor
 torch.set_num_threads(4);torch.set_num_interop_threads(1);cv2.setNumThreads(1)
 torch.cuda.set_per_process_memory_fraction(6*1024**3/torch.cuda.get_device_properties(0).total_memory)
 assert torch.cuda.mem_get_info()[0]>=8*1024**3,'GPU8GiB_free_admission'
 provenance=json.loads((root/'third_party/sam2_color_audit_20261025/RSGG_PROVENANCE.json').read_text())
 for p,h in provenance['source_files_sha256'].items():assert sha(root/'third_party/sam2_color_audit_20261025'/p)==h
 weights=Path.home()/'etechs/thesis/var/models/sam2/sam2.1_hiera_small.pt';assert sha(weights)=='6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38'
 clipdir=Path.home()/'shared-assets/models/clip-vit-base-patch32';clipweights=[p for p in clipdir.iterdir() if p.suffix in ['.bin','.safetensors']]
 assert len(clipweights)==1 and sha(clipweights[0])=='a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f'
 rules=json.loads((root/'configs/vg_object_color_rules_v3_20261025.json').read_text());binding_cfg=json.loads((root/'configs/vg_color_object_binding_native_v1_20261025.json').read_text())
 table=root/'artifacts/color_research_v3_20261025/w2c.mat';parent=json.loads((root/'artifacts/color_rules_v4_train_pilot_20261025/preregistration.json').read_text());assert sha(table)==parent['ColorNames_table_sha256']
 predictor=SAM2ImagePredictor(build_sam2('configs/sam2.1/sam2.1_hiera_s.yaml',str(weights),apply_postprocessing=False),max_hole_area=0,max_sprinkle_area=0)
 clip=ClipVerifier(clipdir);binding=NeutralBinding(clip,rules['class_rules'],binding_cfg);namer=MemoColorNames(table)
 engine=ColorEngine(predictor,clip,binding,namer,rules,'safe_cache');guard.check()
 return engine,torch
class ImageZIPs:
 def __init__(self,root,ids):
  self.root=Path(root);self.wanted=set(ids)
  self.archives=[zipfile.ZipFile(Path.home()/'sbir/hienlth/vg'/n) for n in ['images.zip','images2.zip']];self.members={}
  for z in self.archives:
   for m in z.infolist():
    if not m.filename.lower().endswith('.jpg'):continue
    try:i=int(Path(m.filename).stem)
    except ValueError:continue
    if i in self.wanted:assert i not in self.members;self.members[i]=(z,m)
  assert set(self.members)==self.wanted,'image_ZIP_membership_incomplete_no_denominator_drop'
  with sqlite3.connect((self.root/'artifacts/caption_readiness/vg_pixel_fingerprints_v1.sqlite').as_uri()+'?mode=ro',uri=True) as db:
   self.pixels={i:(h,w,hh) for i,h,w,hh in db.execute('SELECT vg_id,jpeg_sha,width,height FROM pixels') if i in self.wanted}
  assert set(self.pixels)==self.wanted
 def read(self,i):
  from PIL import Image
  z,m=self.members[i];blob=z.read(m);h,w,hh=self.pixels[i];assert hashlib.sha256(blob).hexdigest()==h
  image=Image.open(io.BytesIO(blob)).convert('RGB');assert image.size==(w,hh);return image
 def close(self):
  for z in self.archives:z.close()
def difference(a,b,tol=1e-8,path=''):
 import numpy as np,math
 paths=[];maximum=0.
 if isinstance(a,dict) and isinstance(b,dict):
  if set(a)!=set(b):return [path+':keys'],0.
  for k in a:
   d,m=difference(a[k],b[k],tol,path+'/'+k);paths+=d;maximum=max(maximum,m)
 elif isinstance(a,(list,tuple)) and isinstance(b,(list,tuple)):
  if len(a)!=len(b):return [path+':length'],0.
  for i,(x,y) in enumerate(zip(a,b)):
   d,m=difference(x,y,tol,path+'/'+str(i));paths+=d;maximum=max(maximum,m)
 elif isinstance(a,(float,np.floating)) or isinstance(b,(float,np.floating)):
  if a is None or b is None:return ([path] if a!=b else []),0.
  maximum=abs(float(a)-float(b))
  if not math.isfinite(maximum) or maximum>tol:paths.append(path)
 elif a!=b:paths.append(path)
 return paths,maximum
