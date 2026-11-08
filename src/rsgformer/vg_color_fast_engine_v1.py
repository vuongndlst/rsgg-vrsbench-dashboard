"""Engineering-only elimination of redundant work; original rules remain authoritative."""
import hashlib,math,time
import numpy as np
from PIL import Image
from .vg_object_color_rules_v3 import ColorNames,clipped_box,jitter_boxes,grabcut_mask,apply_rules
from .vg_object_color_v2 import PALETTE
from .sam_binary_mask_adapter_v1 import binary_mask

class MemoColorNames(ColorNames):
 def reset(self):self.cache={};self.bytes=0;self.hits=0
 def summary(self,rgb):
  a=np.ascontiguousarray(rgb);key=(a.shape,a.dtype.str,hashlib.sha256(a.tobytes()).digest())
  if key in self.cache:self.hits+=1;return self.cache[key]
  value=super().summary(a);size=value['labels'].nbytes
  if self.bytes+size<=64*1024**2:self.cache[key]=value;self.bytes+=size
  return value

def children_for(graph,node):
 box=np.asarray(node['bbox_xyxy'],dtype=np.float32);area=max(1,(box[2]-box[0])*(box[3]-box[1]));group=node['surface_group']
 classes={
  'container_contents':set('food fruit vegetable banana orange pizza flower'.split()),
  'vehicle_body':set('wheel tire window windshield light'.split()),
  'garment_or_accessory':set('arm hand head face hair'.split()),
  'animal_coat':set('person man woman boy girl hat helmet'.split())}.get(group,set())
 children=[]
 for n in graph['nodes']:
  b=np.asarray(n['bbox_xyxy']);aa=max(1,(b[2]-b[0])*(b[3]-b[1]));inter=np.maximum(0,np.minimum(b[2:],box[2:])-np.maximum(b[:2],box[:2])).prod()
  if n['label'] in classes and inter/aa>=.5 and aa/area<=.8:children.append(n)
 return sorted(children,key=lambda n:(-n['object_score'],n['id']))[:3]

def early_out(node,box,W,H,children):
 group=node['surface_group']
 if group in ['person_or_skin','transparent_or_emissive','text_symbol']:
  return {'status':'not_requested','reason':'class_surface_rule'}
 if group in ['container_contents','vehicle_body']:
  if not children:return {'status':'unknown','reason':'surface_requires_independent_child_part_exclusion'}
  # Child-quality determines original rule priority; do not skip that computation.
  return None
 b,visible=clipped_box(box,W,H)
 if b is None:info={'reason':'invalid_bbox'}
 elif min(b[2]-b[0],b[3]-b[1])<32:info={'reason':'small_bbox'}
 elif visible<.90:info={'reason':'severely_clipped_bbox','visible_fraction':visible}
 else:return None
 return {'status':'unknown','reason':info['reason'],'mask_diagnostics':[info]}

def binding_from_features(binding,features,target):
 values=(features@binding.text.T).cpu().numpy();family={target};labels=binding.labels;config=binding.config
 for group in config['allowed_label_families']:
  if target in group:family.update(group)
 views=[]
 for vector in values:
  order=np.argsort(-vector);positions={labels[int(j)]:rank+1 for rank,j in enumerate(order)}
  best=max(family,key=lambda name:float(vector[labels.index(name)]));gap=float(vector[int(order[0])]-vector[labels.index(best)])
  views.append({'top5':[{'class':labels[int(j)],'cosine':float(vector[j])} for j in order[:5]],'family_member':best,'family_rank':positions[best],'gap_to_best':gap})
 passed={p['id']:all(v['family_rank']<=p['family_rank_max'] and v['gap_to_best']<=p['maximum_gap_to_best'] for v in views) for p in config['guard_profiles']}
 return {'native_view_order':['original_bbox','mask_fill127','mask_fill255'],'views':views,'profile_pass':passed,'calibrated_probability':False,'human_verified':False}

def shared_clip(clip,binding,original,views,label):
 # Text banks are warmed identically outside the measured trials.
 t=clip.torch;images=[original]+views
 with t.inference_mode():
  f=clip.model.get_image_features(**clip.processor(images=images,return_tensors='pt').to('cuda'))
  if not isinstance(f,t.Tensor):f=f.pooler_output
  f=t.nn.functional.normalize(f.float(),dim=-1)
  values=(f[1:]@clip.text[label].T).cpu().tolist()
  neutral=binding_from_features(binding,f,label)
 rows=[]
 for vector in values:
  order=sorted(range(len(PALETTE)),key=lambda i:-vector[i]);i,j=order[:2]
  rows.append({'top_color':PALETTE[i],'margin':vector[i]-vector[j],'neutral_object_similarity':vector[-1],'color_similarities':dict(zip(PALETTE,vector[:-1]))})
 return rows,neutral

class ColorEngine:
 def __init__(self,predictor,clip,binding,namer,rules,mode):
  assert mode in ['baseline','safe_cache','shared_clip']
  self.predictor=predictor;self.clip=clip;self.binding=binding;self.namer=namer;self.rules=rules;self.mode=mode
  self.current=None;self.encoded=False;self.child_cache={};self.counters={};self.times={}
 def set_image(self,image,identity):
  self.image=image;self.identity=identity;self.rgb=np.asarray(image);self.encoded=False;self.child_cache={}
  if self.mode=='baseline':self.encode()
 def encode(self):
  if not self.encoded:
   start=time.perf_counter();t=self.clip.torch
   with t.inference_mode(),t.autocast('cuda',dtype=t.bfloat16):self.predictor.set_image(self.image)
   t.cuda.synchronize();self.times['SAM_image']=self.times.get('SAM_image',0)+time.perf_counter()-start
   self.encoded=True;self.counters['image_encodes']=self.counters.get('image_encodes',0)+1
 def run(self,target,graph):
  t=self.clip.torch;box=np.asarray(target['node']['bbox_xyxy'],dtype=np.float32);node=dict(target['node']);node['surface_group']=self.rules['class_rules'][node['label']]['surface_group']
  rgb=self.rgb;H,W=rgb.shape[:2];children=children_for(graph,node);fast=self.mode!='baseline'
  if fast:self.namer.reset()
  known=early_out(node,box,W,H,children) if fast else None
  # Preserve the legacy child-exclusion evidence when any child exists.
  if known is not None and children:known=None
  child_ids=[];inner=grab_inner=None;binding=clip_evidence=None
  if known is not None:
   sam=dict(known);grab=dict(known);self.counters['early_out']=self.counters.get('early_out',0)+1
  else:
   self.encode();sam_masks=[];qualities=[];start=time.perf_counter()
   with t.inference_mode(),t.autocast('cuda',dtype=t.bfloat16):
    for b in jitter_boxes(box,W,H):
     masks,quality,_=self.predictor.predict(box=b,multimask_output=False,return_logits=True)
     sam_masks.append(masks[0]>0);qualities.append(float(quality[0]))
   exclude=None
   for child in children:
    key=(child['id'],tuple(child['bbox_xyxy']))
    if fast and key in self.child_cache:
     mask,quality=self.child_cache[key];self.counters['child_cache_hits']=self.counters.get('child_cache_hits',0)+1
    else:
     with t.inference_mode(),t.autocast('cuda',dtype=t.bfloat16):m,q,_=self.predictor.predict(box=np.asarray(child['bbox_xyxy'],dtype=np.float32),multimask_output=False)
     mask,quality=binary_mask(m[0]),float(q[0])
     if fast:self.child_cache[key]=(mask,quality)
    if quality>=.80:exclude=mask if exclude is None else exclude|mask;child_ids.append(child['id'])
   t.cuda.synchronize();self.times['SAM_prompts']=self.times.get('SAM_prompts',0)+time.perf_counter()-start;start=time.perf_counter()
   sam,inner=apply_rules(rgb,sam_masks,box,self.namer,self.rules,node['surface_group'],qualities,exclude)
   grab_masks=[grabcut_mask(rgb,b,target['key']+':'+str(i)) for i,b in enumerate(jitter_boxes(box,W,H))]
   grab,grab_inner=apply_rules(rgb,grab_masks,box,self.namer,self.rules,node['surface_group'],None,exclude)
   self.times['pixel_rules_and_GrabCut']=self.times.get('pixel_rules_and_GrabCut',0)+time.perf_counter()-start
   if inner is not None:
    x0,y0,x1,y1=[max(0,int(v)) for v in box];x1=min(W,x1);y1=min(H,y1);views=[]
    for fill in [127,255]:
     view=np.full_like(rgb,fill);view[inner]=rgb[inner];views.append(Image.fromarray(view[y0:y1,x0:x1]))
    original=self.image.crop([x0,y0,x1,y1]);start=time.perf_counter()
    if self.mode=='shared_clip':clip_evidence,binding=shared_clip(self.clip,self.binding,original,views,node['label'])
    else:clip_evidence=self.clip.scores(views,node['label']);binding=self.binding.score([original]+views,node['label'])
    t.cuda.synchronize();self.times['CLIP']=self.times.get('CLIP',0)+time.perf_counter()-start
  def mask_hash(m):return hashlib.sha256(m.tobytes()).hexdigest() if m is not None else None
  return {'key':target['key'],'methods':{'SAM2_ColorNames':sam,'GrabCut_ColorNames':grab},'CLIP_fill127_fill255':clip_evidence,
   'native_object_binding':binding,'excluded_child_node_ids':child_ids,
   'mask_sha256':{'SAM2_inner':mask_hash(inner),'GrabCut_inner':mask_hash(grab_inner)}}
