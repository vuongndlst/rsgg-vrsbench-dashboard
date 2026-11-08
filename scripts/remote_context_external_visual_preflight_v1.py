"""Read-only sample and resource admission for six external qualitative inferences."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,random,resource,shutil,subprocess
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});resource.setrlimit(resource.RLIMIT_AS,(1024**3,1024**3))
from PIL import Image
root=Path.home()/'rsg-former';shared=Path.home()/'shared-assets/datasets'
raw=(shared/'coco_karpathy/test_manifest.json').read_bytes();assert hashlib.sha256(raw).hexdigest()=='d18d88683e801104b19fd3cc3b1dc6d14d99feb27b71a89cc3dcbcb746f658b4'
metadata=json.loads(raw);assert metadata['metadata_sha256']=='2fd999220673258012acfb411a4e7e66af7d488050b2519b0badcc49b7600b8d'
coco=sorted([(int(x['id']),x['file_name']) for x in metadata['images']]);del metadata,raw
available_coco=[(i,n) for i,n in coco if (shared/'coco_karpathy/images'/n).is_file()]
raw=(shared/'openimages_test/test_identity_manifest.json').read_bytes();assert hashlib.sha256(raw).hexdigest()=='6eb586b40bd68245e20205d896a5ae49700b72d61783b0e5ddc80a50e2242c87'
oi=json.loads(raw);oi_ids=set(oi['caption_test_ids']);available_oi=sorted(p.stem for p in (shared/'openimages_test/images').glob('*.jpg') if p.stem in oi_ids)
records=[]
for dataset,pool in [('coco',available_coco),('openimages',available_oi)]:
 chosen=random.Random(1729).sample(pool,3)
 for item in chosen:
  identity,name=item if dataset=='coco' else (item,item+'.jpg')
  p=shared/('coco_karpathy' if dataset=='coco' else 'openimages_test')/'images'/name
  with Image.open(p) as im:width,height=im.size;fmt=im.format
  assert fmt=='JPEG' and width*height<=16_000_000 and p.stat().st_size<=8*1024**2
  records.append({'dataset':dataset,'split':'test','image_id':identity,'filename':name,'width':width,'height':height,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
q=subprocess.run(['nvidia-smi','--query-gpu=index,memory.total,memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10);assert q.returncode==0
gpu=[{'index':int(v[0]),'total_mib':int(v[1]),'free_mib':int(v[2])} for v in (line.split(',') for line in q.stdout.splitlines())]
free=shutil.disk_usage(root).free
out={'utc':datetime.now(timezone.utc).isoformat(),'read_only':True,'selection_seed':1729,'selection':'3random test images per dataset from sorted existing eligible local-image pool; no GT/prediction-based selection',
     'pools':{'coco_test_total':len(coco),'coco_test_existing':len(available_coco),'oi_protected_test_total':len(oi_ids),'oi_test_existing':len(available_oi)},'samples':records,
     'disk_free_bytes':free,'disk_admission':free>=16*1024**3,'gpus':gpu,'gpu_admission':gpu[0]['free_mib']>=8*1024,
     'new_inference':False,'test_metrics_computed':False}
print(json.dumps(out))
