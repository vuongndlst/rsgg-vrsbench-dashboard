"""TRAIN/VAL-only candidate color sidecar. Accepted attributes require visual gate."""
from pathlib import Path
from datetime import datetime,timezone
import gc,gzip,hashlib,io,json,os,resource,shutil,signal,sqlite3,time,zipfile
os.environ.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',CUDA_CACHE_DISABLE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
ROOT=Path.home()/'rsg-former';BASE=ROOT/'artifacts/vg_caption_execution_v1_20261025';ART=BASE/'color_candidates_v2';START=time.time()
os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:4]));signal.alarm(3600)
from rsgformer.vg_caption_graph_v1 import hash_order,digest
from rsgformer.vg_object_color_v2 import ClipVerifier,prepare_crop,candidates,PALETTE,OPERATING_POINTS,pixel_labels

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def save(path,value):
 tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)
def guard():
 assert time.time()<1794670321.2414355 and time.time()-START<3600
 assert shutil.disk_usage(ROOT).free>=16*1024**3,'reserve16GiB'
 assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=4*1024**3,'rss4GiB'
 assert not (BASE/'STOP').exists(),'owned_stop'
 assert sum(p.stat().st_size for p in BASE.rglob('*') if p.is_file())<=8*1024**3,'new_phase_artifacts8GiB'
def event(stage,**kw):
 guard();save(ART/'progress.json',{'utc':datetime.now(timezone.utc).isoformat(),'stage':stage,**kw})
def main():
 assert (BASE/'base_cache_complete.json').exists() and not ART.exists()
 guard();ART.mkdir();(ART/'thumbnails').mkdir()
 lock={'stage':'candidate_color_cache_v2','source_hashes':{p:sha(ROOT/p) for p in ['scripts/remote_vg_caption_color_cache_v2.py','src/rsgformer/vg_object_color_v2.py','src/rsgformer/vg_caption_graph_v1.py']},
  'palette':PALETTE,'operating_points':OPERATING_POINTS,'node_cap_per_image':12,'train_crop_cap':200,
  'TRAIN_VAL_only':True,'test_accessed':False,'mask_padding':.15,'mask_max_side':256,'grabcut_iterations':5,
  'baseline_independence':'E0/E2 keep raw-crop evidence even if GrabCut quality fails; E1/E3/E4 abstain',
  'acceptance_gate':'candidate only; requires held-out visual precision>=.90 and>=100audited accepted check instances; otherwise no-color default'}
 save(ART/'preregistration.json',lock)
 membership=json.loads((BASE/'membership.json').read_text());parts={i:p for p,ids in membership.items() for i in ids}
 records={}
 for name in ['train_bank_base_sg.jsonl.gz','val_base_sg.jsonl.gz']:
  with gzip.open(BASE/name,'rt') as f:
   for line in f:
    r=json.loads(line)
    if r['image_id'] in parts:records[r['image_id']]=r
 target_ids={i:set() for i in records}
 with gzip.open(BASE/'region_units_no_color.jsonl.gz','rt') as f:
  for line in f:
   u=json.loads(line);target_ids[u['parent_image_id']].update(n['id'] for n in u['graph']['nodes'])
 classes=set('shirt jacket pant short coat hat cap helmet shoe bag bus truck car train cat dog horse cow plate toilet'.split())
 targets=[]
 for identity,record in records.items():
  ns=[n for n in record['graph']['nodes'] if n['id'] in target_ids[identity] and n['label'] in classes]
  ns=sorted(ns,key=lambda n:(-n['object_score'],n['id']))[:12]
  targets.extend({'image_id':identity,'partition':parts[identity],'node':n,'key':str(identity)+':'+n['id']} for n in ns)
 train=sorted([t for t in targets if t['partition']=='train_bank'],key=lambda t:hash_order('VG_COLOR_TRAIN_PROTOTYPE_V2',t['key']))[:200]
 targets=train+sorted([t for t in targets if t['partition']!='train_bank'],key=lambda t:(t['image_id'],t['key']))
 save(ART/'target_node_manifest.json',targets)
 import numpy as np,torch,cv2
 from PIL import Image,ImageOps
 torch.set_num_threads(4);torch.set_num_interop_threads(1);cv2.setNumThreads(1)
 torch.cuda.set_per_process_memory_fraction(6*1024**3/torch.cuda.get_device_properties(0).total_memory)
 assert torch.cuda.mem_get_info()[0]>=8*1024**3,'free_gpu8GiB'
 # Synthetic pixel fixture: background replacement must never count as object black.
 assert pixel_labels(np.asarray([[255,0,0]]))[0][0]=='red'
 assert pixel_labels(np.asarray([[0,0,0]]))[0][0]=='black'
 assert pixel_labels(np.asarray([[255,255,255]]))[0][0]=='white'
 clip=ClipVerifier(Path.home()/'shared-assets/models/clip-vit-base-patch32')
 archives=[zipfile.ZipFile(Path.home()/'sbir/hienlth/vg'/name) for name in ['images.zip','images2.zip']]
 wanted={t['image_id'] for t in targets};members={}
 for z in archives:
  for m in z.infolist():
   if not m.filename.lower().endswith('.jpg'):continue
   try:identity=int(Path(m.filename).stem)
   except ValueError:continue
   if identity in wanted:assert identity not in members;members[identity]=(z,m)
 assert set(members)==wanted
 db=sqlite3.connect((ROOT/'artifacts/caption_readiness/vg_pixel_fingerprints_v1.sqlite').as_uri()+'?mode=ro',uri=True)
 source={i:(h,w,hh) for i,h,w,hh in db.execute('SELECT vg_id,jpeg_sha,width,height FROM pixels') if i in wanted};db.close()
 current=None;image=None;summary={};count=0
 with gzip.open(ART/'color_candidates.jsonl.gz','xt',encoding='utf-8') as out:
  for target in targets:
   guard();identity=target['image_id'];node=target['node']
   if current!=identity:
    if image:image.close()
    z,m=members[identity];raw=z.read(m);assert hashlib.sha256(raw).hexdigest()==source[identity][0]
    image=Image.open(io.BytesIO(raw)).convert('RGB');assert image.size==(source[identity][1],source[identity][2]);current=identity
   crop=prepare_crop(image,node['bbox_xyxy'],target['key'])
   row={'key':target['key'],'image_id':identity,'partition':target['partition'],'node_id':node['id'],'class':node['label'],
    'source_bbox':node['bbox_xyxy'],'bbox_sha256':digest(node['bbox_xyxy']),'image_jpeg_sha256':source[identity][0],
    'status':crop['status'],'attribute_status':'CANDIDATE_ONLY_NOT_VISUALLY_ACCEPTED'}
   if crop.get('raw_image') is not None:
    images=[crop['raw_image']]+([crop['masked_image']] if crop.get('masked_image') else [])
    scores=clip.scores(images,node['label']);raw_clip=scores[0];masked_clip=scores[1] if len(scores)>1 else None
    row.update(candidates=candidates(crop,raw_clip,masked_clip),raw_clip=raw_clip,masked_clip=masked_clip,
     raw_histogram=crop['raw_histogram'],masked_histogram=crop['masked_histogram'],
     foreground_fraction=crop.get('foreground_fraction'),foreground_pixels=crop.get('foreground_pixels'))
    # Derived thumbnail evidence only, not original VG image extraction.
    left=ImageOps.contain(crop['raw_image'],(160,160));right=ImageOps.contain(crop.get('masked_image') or crop['raw_image'],(160,160))
    sheet=Image.new('RGB',(320,160),'white');sheet.paste(left,(0,0));sheet.paste(right,(160,0))
    filename=target['key'].replace(':','_')+'.jpg';sheet.save(ART/'thumbnails'/filename,quality=85);row['thumbnail']=filename
   out.write(json.dumps(row)+'\n');summary[row['status']]=summary.get(row['status'],0)+1;count+=1
   if count%100==0:event('candidate_color_extraction',nodes=count,targets=len(targets),status_counts=summary,peak_allocator_bytes=torch.cuda.max_memory_allocated())
 if image:image.close()
 for z in archives:z.close()
 result={'status':'TRAIN_VAL_CANDIDATE_COLOR_CACHE_COMPLETE_VISUAL_GATE_PENDING','utc':datetime.now(timezone.utc).isoformat(),
  'nodes':count,'images':len(wanted),'status_counts':summary,'color_accepted_for_caption':False,'test_accessed':False,
  'source_sha256':lock['source_hashes'],'color_candidates_sha256':sha(ART/'color_candidates.jsonl.gz'),
  'peak_allocator_bytes':torch.cuda.max_memory_allocated(),'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'wall_seconds':time.time()-START}
 save(ART/'complete.json',result);save(BASE/'color_cache_complete.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':
 try:main()
 except Exception as e:
  if ART.exists():save(ART/'failure.json',{'error_type':type(e).__name__,'reason':str(e),'utc':datetime.now(timezone.utc).isoformat(),'test_accessed':False})
  raise
