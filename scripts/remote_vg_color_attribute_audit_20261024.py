"""Read-only VG150 object-bound color GT statistics on TRAIN/VAL; no TEST attributes."""
from pathlib import Path
from datetime import datetime,timezone
import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import gc,hashlib,json,math,re,resource,signal,time,zipfile
from collections import Counter
ROOT=Path.home()/'rsg-former';START=time.monotonic();os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});signal.alarm(100)
import h5py,numpy as np
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
dictionary=ROOT/'data/vg150/VG-SGG-dicts-with-attri.json';h5=ROOT/'data/vg150/VG-SGG-with-attri.h5';meta=ROOT/'data/vg150/image_data.json'
info=json.loads(dictionary.read_text());objects={int(k):v for k,v in info['idx_to_label'].items()};attributes={int(k):v for k,v in info['idx_to_attribute'].items()};assert set(objects)==set(range(1,151))
images=[x for x in json.loads(meta.read_text()) if x['image_id'] not in {1592,1722,4616,4617}];assert len(images)==108073
ids={};ann_sha={}
with zipfile.ZipFile(ROOT/'data/vg_canonical_audit/reltr_vg_annotations.zip') as z:
 for split in ['train','val']:
  raw=z.read('vg/'+split+'.json');ann_sha[split]=hashlib.sha256(raw).hexdigest();data=json.loads(raw);ids[split]={int(i['id']) for i in data['images']};del raw,data;gc.collect()
assert len(ids['train'])==57723 and len(ids['val'])==5000 and not ids['train']&ids['val']
core={'black','white','red','green','blue','yellow','brown','gray','orange','pink','purple','beige','tan'}
aliases={'grey':'gray','blond':'blond','blonde':'blond','golden':'gold','gold':'gold','silver':'silver'}
def color_labels(text):
 tokens=re.findall('[a-z]+',text.lower());return sorted(set(t if t in core else aliases[t] for t in tokens if t in core or t in aliases))
mapping={i:color_labels(t) for i,t in attributes.items()};color_attrs={i:{'attribute':attributes[i],'colors':c} for i,c in mapping.items() if c}
non_exact_color_attrs={i:row for i,row in color_attrs.items() if row['attribute'].lower() not in core|set(aliases)}
def fresh():return {'objects':0,'with_any_attribute':0,'with_color':0,'with_multiple_color_labels':0,'images':0,'images_with_color':0,'colors':Counter(),'raw_color_attributes':Counter()}
stats={s:{i:fresh() for i in objects} for s in ids};source_image_counts=Counter();h5_schema={}
with h5py.File(h5,'r') as f:
 h5_schema={k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in f.items()};assert 'attributes' in f
 first=f['img_to_first_box'][:];last=f['img_to_last_box'][:];labels=f['labels'][:,0];atts=f['attributes']
 assert len(first)==len(images) and atts.shape[0]==len(labels)
 # Only in-scope images have their object-attribute slices read.
 for index,image in enumerate(images):
  identity=int(image['image_id']);split='train' if identity in ids['train'] else 'val' if identity in ids['val'] else None
  if split is None:continue
  a,b=int(first[index]),int(last[index])+1;assert a>=0 and b>a;source_image_counts[split]+=1;seen=set();seen_color=set()
  local_labels=labels[a:b];local_atts=atts[a:b]
  for cls,row in zip(local_labels,local_atts):
   cls=int(cls);assert cls in objects;seen.add(cls);st=stats[split][cls];st['objects']+=1;values={int(v) for v in row if int(v)>0};assert all(v in attributes for v in values)
   st['with_any_attribute']+=bool(values);colors=set()
   for v in values:
    if mapping[v]:colors.update(mapping[v]);st['raw_color_attributes'][attributes[v]]+=1
   if colors:
    st['with_color']+=1;st['with_multiple_color_labels']+=len(colors)>1;st['colors'].update(colors);seen_color.add(cls)
  for cls in seen:stats[split][cls]['images']+=1
  for cls in seen_color:stats[split][cls]['images_with_color']+=1
  if time.monotonic()-START>80:raise RuntimeError('bounded_audit_time_cap_no_partial_ranking')
assert dict(source_image_counts)=={'val':5000,'train':57723} or dict(source_image_counts)=={'train':57723,'val':5000}
out={}
for split,classes in stats.items():
 rows=[]
 for cls,st in classes.items():
  n=st['objects'];k=st['with_color'];rate=k/n if n else None;z=1.96
  lower=((rate+z*z/(2*n))-z*math.sqrt((rate*(1-rate)+z*z/(4*n))/n))/(1+z*z/n) if n else None
  counts=st['colors'];total=sum(counts.values());entropy=-sum((c/total)*math.log2(c/total) for c in counts.values()) if total else None
  rows.append({'class_id':cls,'class_name':objects[cls],**st,'color_rate_all_objects':rate,'color_rate_among_attribute_annotated':k/st['with_any_attribute'] if st['with_any_attribute'] else None,'color_rate_wilson_lower95':lower,'color_label_entropy_bits':entropy,'top_colors':counts.most_common(8)})
 out[split]=rows
runtime={}
for name in ['cv2','transformers']:
 try:
  module=__import__(name);runtime[name]=getattr(module,'__version__','installed')
 except ImportError:runtime[name]=None
clip=Path.home()/'shared-assets/models/clip-vit-base-patch32';clip_files={p.name:{'bytes':p.stat().st_size} for p in clip.iterdir() if p.is_file()} if clip.is_dir() else {}
print(json.dumps({'status':'VG150_TRAIN_VAL_OBJECT_COLOR_AUDIT_COMPLETE','utc':datetime.now(timezone.utc).isoformat(),'read_only':True,'server_files_written':0,'model_loaded':False,'image_pixels_read':False,'test_attribute_semantics_analyzed':False,'caption_test_scored':False,'provenance':{'dictionary_sha256':sha(dictionary),'h5_sha256':sha(h5),'image_metadata_sha256':sha(meta),'canonical_annotation_sha256':ann_sha},'h5_schema':h5_schema,'color_attribute_mapping':color_attrs,'non_exact_color_attribute_names_requiring_review':non_exact_color_attrs,'attribute_names':attributes,'normalization':'grey->gray; blonde/blond->blond; golden/gold->gold; multiword color-bearing attribute tokens retained and flagged, not semantic color truth verification','denominator':'all canonical in-scope H5 object instances; missing attribute is unlabeled, not negative color GT','splits':out,'runtime_versions':runtime,'existing_shared_clip_files':clip_files,'wall_seconds':time.monotonic()-START,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}))
