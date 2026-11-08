"""Frozen200 TRAIN-node engineering pilot; new masks/rules, no VAL-check/TEST tuning."""
from pathlib import Path
from datetime import datetime,timezone
import collections,gc,gzip,hashlib,io,json,math,os,resource,shutil,signal,sqlite3,subprocess,sys,time,traceback,zipfile
os.environ.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUDA_CACHE_DISABLE='1')
ROOT=Path.home()/'rsg-former';BASE=ROOT/'artifacts/vg_caption_execution_v1_20261025';ART=ROOT/'artifacts/color_rules_v4_train_pilot_20261025';START=time.time()
sys.path.insert(0,str(ROOT/'third_party/sam2_color_audit_20261025'))
os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:4]));signal.alarm(1800)
import cv2,numpy as np,torch
from PIL import Image,ImageOps
from rsgformer.vg_object_color_rules_v3 import ColorNames,apply_rules,jitter_boxes,grabcut_mask,NAMES
from rsgformer.vg_object_color_v2 import ClipVerifier
from rsgformer.sam_binary_mask_adapter_v1 import binary_mask
from rsgformer.vg_color_binding_native_v1 import NeutralBinding
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,indent=2));t.replace(p)
def guard():
 assert time.time()<1794670321.2414355 and time.time()-START<1800,'wall_deadline'
 assert shutil.disk_usage(ROOT).free>=16*1024**3,'disk_reserve16GiB'
 assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=4*1024**3,'RSS4GiB'
 assert not (ART/'STOP').exists() and not (BASE/'STOP').exists(),'owned_STOP'
 q=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
 used=sum(int(line.split(',')[1])*1024**2 for line in q.splitlines() if int(line.split(',')[0])==os.getpid())
 assert used<=8*1024**3,'own_GPU8GiB'
def main():
 assert not ART.exists();guard();ART.mkdir();(ART/'thumbnails').mkdir()
 parent=ROOT/'artifacts/color_rules_v3_train_pilot_r1_20261025';assert (parent/'complete.json').exists()
 save(ART/'parent_provenance.json',{'parent_phase':str(parent),'complete_sha256':sha(parent/'complete.json'),'change':'native-resolution neutral class/family guard, replacing diagnostic grid-thumbnail binding view; masks/color thresholds unchanged; no VAL/TEST access'})
 rules=json.loads((ROOT/'configs/vg_object_color_rules_v3_20261025.json').read_text())
 table=ROOT/'artifacts/color_research_v3_20261025/w2c.mat';weights=Path.home()/'etechs/thesis/var/models/sam2/sam2.1_hiera_small.pt'
 assert sha(weights)=='6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38'
 provenance=json.loads((ROOT/'third_party/sam2_color_audit_20261025/RSGG_PROVENANCE.json').read_text())
 for name,h in provenance['source_files_sha256'].items():assert sha(ROOT/'third_party/sam2_color_audit_20261025'/name)==h
 targets=[t for t in json.loads((BASE/'color_candidates_v2/target_node_manifest.json').read_text()) if t['partition']=='train_bank'];assert len(targets)==200
 targets=sorted(targets,key=lambda r:(r['image_id'],r['key']))
 lock={'phase':'color_rules_v4_train_pilot_20261025','created_utc':datetime.now(timezone.utc).isoformat(),'fixed_target_nodes':200,'TRAIN_only':True,
  'source_sha256':{p:sha(ROOT/p) for p in ['scripts/remote_vg_color_rules_v4_train_pilot_20261025.py','src/rsgformer/vg_color_binding_native_v1.py','configs/vg_color_object_binding_native_v1_20261025.json','src/rsgformer/sam_binary_mask_adapter_v1.py','src/rsgformer/vg_object_color_rules_v3.py','src/rsgformer/vg_object_color_v2.py','configs/vg_object_color_rules_v3_20261025.json']},
  'SAM2_commit':provenance['commit'],'SAM2_source_manifest_sha256':sha(ROOT/'third_party/sam2_color_audit_20261025/RSGG_PROVENANCE.json'),
  'SAM2_weights_sha256':sha(weights),'ColorNames_table_sha256':sha(table),'original_target_manifest_sha256':sha(BASE/'color_candidates_v2/target_node_manifest.json'),
  'operating_points':rules['finite_operating_points'],'method_variants':rules['mask_variants'],'maximum_wall_seconds':1800,
  'limits':{'own_GPU_GiB':8,'allocator_GiB':6,'RSS_GiB':4,'CPU':4,'reserve_GiB':16},'TEST_access_allowed':False,
  'production_accepted_colors':False,'human_reviews_completed':0,'no_new_threshold_selection_from_check':True}
 save(ART/'preregistration.json',lock);save(ART/'target_manifest.json',targets)
 namer=ColorNames(table)
 for rgb,expected in [([0,0,0],'black'),([255,255,255],'white'),([255,0,0],'red'),([255,190,200],'pink'),([130,20,20],'red')]:
  assert namer.summary(np.asarray([rgb],dtype=np.uint8))['order'][0]==expected,'learned_LUT_synthetic_contract'
 torch.set_num_threads(4);torch.set_num_interop_threads(1);cv2.setNumThreads(1)
 torch.cuda.set_per_process_memory_fraction(6*1024**3/torch.cuda.get_device_properties(0).total_memory)
 assert torch.cuda.mem_get_info()[0]>=8*1024**3
 from sam2.build_sam import build_sam2
 from sam2.sam2_image_predictor import SAM2ImagePredictor
 predictor=SAM2ImagePredictor(build_sam2('configs/sam2.1/sam2.1_hiera_s.yaml',str(weights),apply_postprocessing=False),max_hole_area=0,max_sprinkle_area=0)
 clip=ClipVerifier(Path.home()/'shared-assets/models/clip-vit-base-patch32');binding_cfg=json.loads((ROOT/'configs/vg_color_object_binding_native_v1_20261025.json').read_text());binding_guard=NeutralBinding(clip,rules['class_rules'],binding_cfg);guard()
 wanted={t['image_id'] for t in targets};archives=[zipfile.ZipFile(Path.home()/'sbir/hienlth/vg'/n) for n in ['images.zip','images2.zip']];members={}
 for archive in archives:
  for member in archive.infolist():
   if not member.filename.lower().endswith('.jpg'):continue
   try:identity=int(Path(member.filename).stem)
   except ValueError:continue
   if identity in wanted:assert identity not in members;members[identity]=(archive,member)
 assert set(members)==wanted
 with sqlite3.connect((ROOT/'artifacts/caption_readiness/vg_pixel_fingerprints_v1.sqlite').as_uri()+'?mode=ro',uri=True) as db:
  pixels={i:(jpeg,w,h) for i,jpeg,w,h in db.execute('SELECT vg_id,jpeg_sha,width,height FROM pixels') if i in wanted}
 with gzip.open(BASE/'train_bank_base_sg.jsonl.gz','rt') as f:graphs={r['image_id']:r['graph'] for r in map(json.loads,f) if r['image_id'] in wanted}
 current=None;image=None;rgb=None;summary=collections.Counter();accepted=collections.Counter();count=0
 with gzip.open(ART/'candidate_records.jsonl.gz','xt') as out:
  for target in targets:
   guard();identity=target['image_id'];node=target['node'];box=np.asarray(node['bbox_xyxy'],dtype=np.float32);rule=rules['class_rules'][node['label']]
   if identity!=current:
    z,m=members[identity];blob=z.read(m);assert hashlib.sha256(blob).hexdigest()==pixels[identity][0]
    image=Image.open(io.BytesIO(blob)).convert('RGB');assert image.size==pixels[identity][1:];rgb=np.asarray(image);current=identity
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):predictor.set_image(image)
   H,W=rgb.shape[:2];boxes=jitter_boxes(box,W,H);sam_masks=[];qualities=[]
   with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
    for b in boxes:
     masks,quality,_=predictor.predict(box=b,multimask_output=False,return_logits=True)
     sam_masks.append(masks[0]>0);qualities.append(float(quality[0]))
   # Remove separate predicted child surfaces, never GT geometry/labels.
   group=rule['surface_group'];child_classes={
    'container_contents':set('food fruit vegetable banana orange pizza flower'.split()),
    'vehicle_body':set('wheel tire window windshield light'.split()),
    'garment_or_accessory':set('arm hand head face hair'.split()),
    'animal_coat':set('person man woman boy girl hat helmet'.split())}.get(group,set())
   children=[];area=max(1,(box[2]-box[0])*(box[3]-box[1]))
   for n in graphs[identity]['nodes']:
    b=np.asarray(n['bbox_xyxy']);aa=max(1,(b[2]-b[0])*(b[3]-b[1]));inter=np.maximum(0,np.minimum(b[2:],box[2:])-np.maximum(b[:2],box[:2])).prod()
    if n['label'] in child_classes and inter/aa>=.5 and aa/area<=.8:children.append(n)
   children=sorted(children,key=lambda n:(-n['object_score'],n['id']))[:3];exclude=None;child_ids=[]
   for child in children:
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):m,q,_=predictor.predict(box=np.asarray(child['bbox_xyxy'],dtype=np.float32),multimask_output=False)
    if float(q[0])>=.80:
     child_mask=binary_mask(m[0]);exclude=child_mask if exclude is None else exclude|child_mask;child_ids.append(child['id'])
   sam,inner=apply_rules(rgb,sam_masks,box,namer,rules,group,qualities,exclude)
   grab_masks=[grabcut_mask(rgb,b,target['key']+':'+str(i)) for i,b in enumerate(boxes)]
   grab,grab_inner=apply_rules(rgb,grab_masks,box,namer,rules,group,None,exclude)
   clip_evidence=None;binding=None
   if inner is not None:
    x0,y0,x1,y1=[max(0,int(v)) for v in box];x1=min(W,x1);y1=min(H,y1)
    views=[]
    for fill in [127,255]:
     view=np.full_like(rgb,fill);view[inner]=rgb[inner];views.append(Image.fromarray(view[y0:y1,x0:x1]))
    clip_evidence=clip.scores(views,node['label'])
    binding=binding_guard.score([image.crop([x0,y0,x1,y1])]+views,node['label'])
   row={'key':target['key'],'image_id':identity,'partition':'TRAIN_engineering_only','class':node['label'],'surface_group':group,
    'bbox_xyxy':node['bbox_xyxy'],'jpeg_sha256':pixels[identity][0],'rules_sha256':lock['source_sha256']['configs/vg_object_color_rules_v3_20261025.json'],
    'methods':{'SAM2_ColorNames':sam,'GrabCut_ColorNames':grab},'CLIP_fill127_fill255':clip_evidence,'excluded_child_node_ids':child_ids,
    'native_object_binding':binding,'unresolved_semantic_surface_review_required':True,'human_reviewed':False,'production_accepted':False}
   for method,res in row['methods'].items():
    summary[method+':'+res['reason']]+=1
    for op,item in res.get('operating_points',{}).items():accepted[method+':'+op]+=bool(item['colors'])
   # Up to200 derived evidence images, no original VG extraction.
   b=[max(0,math.floor(box[0])),max(0,math.floor(box[1])),min(W,math.ceil(box[2])),min(H,math.ceil(box[3]))]
   if b[2]>b[0] and b[3]>b[1]:
    canvas=Image.new('RGB',(480,160),'white');raw=image.crop(b);raw=ImageOps.contain(raw,(160,160));canvas.paste(raw,(0,0))
    for j,mask in enumerate([grab_inner,inner],1):
     view=np.full_like(rgb,127)
     if mask is not None:view[mask]=rgb[mask]
     canvas.paste(ImageOps.contain(Image.fromarray(view).crop(b),(160,160)),(160*j,0))
    filename=target['key'].replace(':','_')+'.jpg';canvas.save(ART/'thumbnails'/filename,quality=90);row['thumbnail']=filename
   out.write(json.dumps(row)+'\n');out.flush();count+=1
   if count%20==0:save(ART/'progress.json',{'utc':datetime.now(timezone.utc).isoformat(),'stage':'TRAIN_mask_color_rule_pilot','committed_nodes':count,'planned_nodes':200,'test_accessed':False})
   gc.collect()
 for z in archives:z.close()
 result={'status':'COLOR_RULES_V4_TRAIN_ENGINEERING_PILOT_COMPLETE_NOT_VALIDATED','utc':datetime.now(timezone.utc).isoformat(),'nodes':count,'images':len(wanted),
  'reason_counts':dict(summary),'candidate_counts':dict(accepted),'candidate_records_sha256':sha(ART/'candidate_records.jsonl.gz'),
  'peak_allocator_bytes':torch.cuda.max_memory_allocated(),'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
  'wall_seconds':time.time()-START,'human_review_count':0,'test_accessed':False,'production_accepted':False}
 save(ART/'complete.json',result);save(ROOT/'reports/VG_COLOR_RULES_V4_TRAIN_PILOT_20261025.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':
 try:main()
 except Exception as error:
  if ART.exists():save(ART/'failure.json',{'error_type':type(error).__name__,'reason':str(error),'test_accessed':False})
  traceback.print_exc();raise
