"""Six fixed qualitative external images. Read-only model execution; stdout only."""
from pathlib import Path
import os
os.environ.update(PYTHONDONTWRITEBYTECODE='1',CUDA_CACHE_DISABLE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
import base64,gc,hashlib,io,json,random,resource,signal,subprocess,time
from datetime import datetime,timezone
ROOT=Path.home()/'rsg-former';SHARED=Path.home()/'shared-assets/datasets';START=time.time()
signal.alarm(180);os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:4]))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
lock_path=ROOT/'artifacts/vg_component_control_test_v1/preregistration.json'
assert sha(lock_path)=='6b57ddbdbaf51c7d9307ae33f9766efc74cfe639cc51b98f9e36adf9dd659b77'
lock=json.loads(lock_path.read_text())
for p,h in lock['source_hashes'].items():assert sha(ROOT/p)==h,p
checkpoint=ROOT/lock['checkpoint'];assert sha(checkpoint)=='29017a69d7cb6d76d68c834b5d540712775a9e0cf9cae13efd17d78a7b69292e'
coco_path=SHARED/'coco_karpathy/test_manifest.json';oi_path=SHARED/'openimages_test/test_identity_manifest.json'
assert sha(coco_path)=='d18d88683e801104b19fd3cc3b1dc6d14d99feb27b71a89cc3dcbcb746f658b4'
assert sha(oi_path)=='6eb586b40bd68245e20205d896a5ae49700b72d61783b0e5ddc80a50e2242c87'
coco=json.loads(coco_path.read_text());coco_ids={int(x['id']) for x in coco['images']};del coco
oi_ids=set(json.loads(oi_path.read_text())['caption_test_ids'])
selected=[('coco',39081,'COCO_val2014_000000039081.jpg','b2c18e086daeccae02e02a94f3ec15a4c236ae113dbfaa4f79bef72f820bd674'),('coco',391940,'COCO_val2014_000000391940.jpg','55780cff2b43259191ab6ea436450ca3d0706b74aba5810728a5af2806b48a99'),('coco',473318,'COCO_val2014_000000473318.jpg','7e8a862da24f0d33683a6c9b8fffb05c43ad68f3cd690f9aa9588a6c16d52ea4'),('openimages','166f70ea6f71c41d','166f70ea6f71c41d.jpg','da7d951e38dc0349c9ffe4a98d488ccc87e538a040972a0727bd0124432427d4'),('openimages','cd842a5d9dcedee8','cd842a5d9dcedee8.jpg','5196d5548ee439c084d67db1b2b3987c17d4309c25398b6fd52522dc4d03d0ec'),('openimages','e3240abe64765a46','e3240abe64765a46.jpg','45d8e223407c89cba4cad15d7fd0c2dee8d8323875eedd39a600030e7171f933')]
q=subprocess.run(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10);assert q.returncode==0 and int(q.stdout.splitlines()[0])>=8192
import numpy as np,torch
from PIL import Image
from rsgformer.vg_v2_runtime import native_model,capture,Adaptation
from rsgformer.vg_native_postprocess import native_vg_postprocess
torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.set_float32_matmul_precision('highest');torch.manual_seed(1729);random.seed(1729);np.random.seed(1729)
torch.cuda.set_per_process_memory_fraction(6*1024**3/torch.cuda.get_device_properties(0).total_memory)
model,_,_=native_model(ROOT);ck=torch.load(checkpoint,map_location='cpu',weights_only=True)
assert ck['variant']=='context' and ck['seed']==1729 and ck['epoch']==10 and ck['microbatch']==2
assert set(ck['native_head'])=={k for k in model.state_dict() if 'add_add_add_relation' in k}
loaded=model.load_state_dict(ck['native_head'],strict=False);assert not loaded.unexpected_keys
adapt=Adaptation(model,'context');adapt.residual.load_state_dict(ck['residual'],strict=True)
for p in adapt.parameters():p.requires_grad_(False)
adapt.cuda().eval();model.eval();del ck;gc.collect()
samples=[];max_gpu=0
for index,(dataset,identity,filename,expected) in enumerate(selected):
 assert identity in (coco_ids if dataset=='coco' else oi_ids)
 path=SHARED/('coco_karpathy' if dataset=='coco' else 'openimages_test')/'images'/filename
 data=path.read_bytes();assert hashlib.sha256(data).hexdigest()==expected
 with Image.open(io.BytesIO(data)) as source:image=source.convert('RGB')
 with torch.no_grad():
  item,full=capture(model,image);del full
  assert all(torch.isfinite(v).all() for v in item.values())
  batch={k:item[k][None].float().cuda() for k in ['nodes','boxes','logits','visual']}
  out,_=adapt(batch);pred=native_vg_postprocess(out,*image.size)
  samples.append({'dataset':dataset,'split':'test','image_id':dataset+'_'+str(identity),'source_image_id':identity,'test_index':index,'image':{'width':image.width,'height':image.height},'prediction':{k:v.tolist() if hasattr(v,'tolist') else v for k,v in pred.items()},'image_jpeg_base64':base64.b64encode(data).decode(),'image_jpeg_sha256':expected})
 own=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10);assert own.returncode==0
 gpu=[int(x.split(',')[1]) for x in own.stdout.splitlines() if int(x.split(',')[0])==os.getpid()];assert len(gpu)==1 and gpu[0]<=8192;max_gpu=max(max_gpu,gpu[0])
 assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=4*1024**3
 del item,batch,out,pred,image,data;gc.collect();torch.cuda.empty_cache()
print(json.dumps({'status':'EXPORTED','utc':datetime.now(timezone.utc).isoformat(),'checkpoint_sha256':lock['checkpoint_sha256'],'sample_indices':list(range(6)),'prediction_artifact_sha256':None,'samples':samples,'new_inference':True,'new_test_scoring':False,'server_files_written':0,'disk_reserve_changed':False,'execution':'explicitly requested zero-write qualitative inference; storage-blocked audit remains paused','max_own_gpu_mib':max_gpu,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'wall_seconds':time.time()-START}))
