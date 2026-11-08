"""Versioned research extractor: mask stability, learned sRGB naming, abstention.

No references, GT attributes, canonical object colors or LLM answers are inputs.
Outputs remain candidates until the independent version/class review gate passes.
"""
import hashlib,math
import cv2
import numpy as np
from scipy.io import loadmat

NAMES='black blue brown gray green orange pink purple red white yellow'.split()

class ColorNames:
 def __init__(self,path):
  self.table=loadmat(path)['w2c'];assert self.table.shape==(32768,11)
  assert np.isfinite(self.table).all() and np.allclose(self.table.sum(1),1,atol=1e-10)
 def probabilities(self,rgb):
  a=np.asarray(rgb,dtype=np.uint8).reshape(-1,3).astype(np.int32)
  i=a[:,0]//8+32*(a[:,1]//8)+1024*(a[:,2]//8)
  return self.table[i]
 def summary(self,rgb):
  values=self.probabilities(rgb);labels=values.argmax(1);mass=np.bincount(labels,minlength=11)/max(1,len(labels))
  mean=values.mean(0);order=sorted(range(11),key=lambda i:(-mass[i],i))
  return {'mass':{NAMES[i]:float(mass[i]) for i in range(11)},'mean_lookup_score':dict(zip(NAMES,map(float,mean))),
   'order':[NAMES[i] for i in order],'labels':labels,'lookup_scores_are_calibrated_confidence':False}

def clipped_box(box,W,H):
 if len(box)!=4 or not all(math.isfinite(v) for v in box) or box[2]<=box[0] or box[3]<=box[1]:return None,0
 raw=(box[2]-box[0])*(box[3]-box[1]);b=[max(0,math.floor(box[0])),max(0,math.floor(box[1])),min(W,math.ceil(box[2])),min(H,math.ceil(box[3]))]
 return b,max(0,b[2]-b[0])*max(0,b[3]-b[1])/max(1,raw)

def jitter_boxes(box,W,H,fraction=.03):
 x0,y0,x1,y1=map(float,box);dx=(x1-x0)*fraction;dy=(y1-y0)*fraction
 return [np.asarray([max(0,x0+sign*dx),max(0,y0+sign*dy),min(W,x1+sign*dx),min(H,y1+sign*dy)],dtype=np.float32) for sign in [0,-1,1]]

def mask_iou(a,b):return float(np.logical_and(a,b).sum()/max(1,np.logical_or(a,b).sum()))

def grabcut_mask(rgb,box,key):
 H,W=rgb.shape[:2];b,visible=clipped_box(box,W,H)
 if b is None or min(b[2]-b[0],b[3]-b[1])<32:return np.zeros((H,W),bool)
 dx=(b[2]-b[0])*.15;dy=(b[3]-b[1])*.15
 p=[max(0,math.floor(b[0]-dx)),max(0,math.floor(b[1]-dy)),min(W,math.ceil(b[2]+dx)),min(H,math.ceil(b[3]+dy))]
 roi=rgb[p[1]:p[3],p[0]:p[2]];s=min(1,256/max(roi.shape[:2]));small=cv2.resize(roi,None,fx=s,fy=s,interpolation=cv2.INTER_AREA)
 x=round((b[0]-p[0])*s);y=round((b[1]-p[1])*s);w=min(round((b[2]-b[0])*s),small.shape[1]-x);h=min(round((b[3]-b[1])*s),small.shape[0]-y)
 result=np.zeros((H,W),bool)
 if min(w,h)<2 or w*h>=small.shape[0]*small.shape[1]:return result
 mask=np.zeros(small.shape[:2],np.uint8);cv2.setRNGSeed(int(hashlib.sha256(key.encode()).hexdigest()[:7],16))
 try:cv2.grabCut(cv2.cvtColor(small,cv2.COLOR_RGB2BGR),mask,(x,y,w,h),np.zeros((1,65)),np.zeros((1,65)),5,cv2.GC_INIT_WITH_RECT)
 except cv2.error:return result
 full=cv2.resize(((mask==1)|(mask==3)).astype(np.uint8),(roi.shape[1],roi.shape[0]),interpolation=cv2.INTER_NEAREST)
 result[p[1]:p[3],p[0]:p[2]]=full>0
 return result

def summarize_mask(rgb,mask,box):
 H,W=mask.shape;b,visible=clipped_box(box,W,H)
 if b is None:return None,{'reason':'invalid_bbox'}
 if min(b[2]-b[0],b[3]-b[1])<32:return None,{'reason':'small_bbox'}
 if visible<.90:return None,{'reason':'severely_clipped_bbox','visible_fraction':visible}
 count=int(mask.sum());inside=np.zeros_like(mask);inside[b[1]:b[3],b[0]:b[2]]=True
 outside=float((mask&~inside).sum()/max(1,count));mask=mask&inside
 area=(b[2]-b[0])*(b[3]-b[1]);count=int(mask.sum());fraction=count/max(1,area)
 if outside>.05:return None,{'reason':'mask_leaks_outside_target_box','outside_fraction':outside}
 if count<256 or fraction<.05:return None,{'reason':'insufficient_mask_pixels','pixels':count,'fraction':fraction}
 _,labels,stats,_=cv2.connectedComponentsWithStats(mask.astype(np.uint8),8)
 largest=max(stats[1:,cv2.CC_STAT_AREA],default=0)/max(1,count)
 if largest<.75:return None,{'reason':'fragmented_target_mask','largest_component_share':float(largest)}
 radius=max(1,min(3,round(.01*min(b[2]-b[0],b[3]-b[1]))))
 inner=cv2.erode(mask.astype(np.uint8),np.ones((2*radius+1,2*radius+1),np.uint8))>0
 if int(inner.sum())<256:return None,{'reason':'too_little_inner_surface'}
 dark=float((rgb[inner].max(1)<=5).mean());bright=float((rgb[inner].min(1)>=250).mean())
 if dark>.60 or bright>.60:return None,{'reason':'severe_clipping_or_underexposure','near_black_clip_fraction':dark,'near_white_clip_fraction':bright}
 return inner,{'reason':'mask_geometry_pass','pixels':int(inner.sum()),'foreground_fraction':fraction,'largest_component_share':float(largest),'outside_fraction':outside,'erosion_radius':radius,
  'near_black_clip_fraction':dark,'near_white_clip_fraction':bright}

def coherent_color_patch(mask,label_map,index):
 target=mask&(label_map==index);n=int(target.sum())
 if not n:return 0
 _,_,stats,_=cv2.connectedComponentsWithStats(target.astype(np.uint8),8)
 return float(max(stats[1:,cv2.CC_STAT_AREA],default=0)/n)

def color_candidate(rgb,inner,namer,point):
 stats=namer.summary(rgb[inner]);a,b=stats['order'][:2];ma,mb=stats['mass'][a],stats['mass'][b]
 scores=stats['mean_lookup_score'];by_score=sorted(scores,key=lambda x:-scores[x]);margin=scores[a]-max(v for k,v in scores.items() if k!=a)
 report={k:v for k,v in stats.items() if k!='labels'};report['lookup_score_margin']=margin
 # Learned color-name outputs are evidence scores, not calibrated correctness probabilities.
 if by_score[0]!=a or margin<point['color_margin_min']:return [],dict(report,reason='ambiguous_color_name')
 chosen=[]
 if ma>=point['primary_mass_min']:chosen=[a]
 elif ma>=point['second_mass_min'] and mb>=point['second_mass_min'] and ma+mb>=point['top2_mass_min']:
  # White/gray or dark-gray/black under shading is not automatically multicolor.
  achromatic={'black','white','gray'}
  if a in achromatic and b in achromatic and {a,b}!={'black','white'}:return [],dict(report,reason='achromatic_shading_ambiguous')
  full_labels=np.full(inner.shape,-1,np.int16);full_labels[inner]=stats['labels']
  labs=cv2.cvtColor(rgb.astype(np.float32)/255,cv2.COLOR_RGB2LAB)
  first=np.median(labs[inner&(full_labels==NAMES.index(a))],axis=0);second=np.median(labs[inner&(full_labels==NAMES.index(b))],axis=0)
  delta=float(np.linalg.norm(first-second));patches=[coherent_color_patch(inner,full_labels,NAMES.index(c)) for c in [a,b]]
  report.update(two_color_Lab_distance=delta,two_color_largest_patch_shares=patches)
  if delta<25 or min(patches)<.50:return [],dict(report,reason='two_colors_not_spatially_coherent')
  chosen=[a,b]
 else:return [],dict(report,reason='no_stable_dominant_color')
 # Sensitivity check only, no white-balance correction or claimed intrinsic reflectance.
 exposure=[]
 for gain in [.9,1.1]:
  shifted=np.clip(rgb[inner].astype(np.float32)*gain,0,255).astype(np.uint8)
  exposure.append(namer.summary(shifted)['order'][0])
 if any(c not in chosen for c in exposure):return [],dict(report,reason='exposure_sensitive_color',exposure_top=exposure)
 return chosen,dict(report,reason='candidate_color_not_yet_visually_validated',exposure_top=exposure,primary_color=chosen[0] if ma>=point['primary_mass_min'] else None)

def apply_rules(rgb,masks,box,namer,rules,group,model_quality=None,part_exclusion=None):
 if group in ['person_or_skin','transparent_or_emissive','text_symbol']:return {'status':'not_requested','reason':'class_surface_rule'},None
 if group in ['container_contents','vehicle_body'] and part_exclusion is None:
  return {'status':'unknown','reason':'surface_requires_independent_child_part_exclusion'},None
 geometric=[];infos=[]
 for mask in masks:
  if part_exclusion is not None:mask=mask&~part_exclusion
  inner,info=summarize_mask(rgb,mask,box);infos.append(info)
  if inner is None:return {'status':'unknown','reason':info['reason'],'mask_diagnostics':infos},None
  geometric.append(inner)
 if model_quality is not None and min(model_quality)<rules['mask_quality']['SAM_predicted_quality_screen_min']:
  return {'status':'unknown','reason':'SAM_predicted_quality_screen','model_quality':list(map(float,model_quality))},geometric[0]
 stability=min(mask_iou(geometric[0],m) for m in geometric[1:]) if len(geometric)>1 else 1
 variants={}
 for point in rules['finite_operating_points']:
  if stability<point['mask_jitter_iou_min']:variants[point['id']]={'colors':[],'reason':'bbox_mask_instability'};continue
  evaluations=[color_candidate(rgb,m,namer,point) for m in geometric]
  colors=evaluations[0][0]
  if not colors or any(set(c)!=set(colors) for c,d in evaluations[1:]):colors=[]
  variants[point['id']]={'colors':colors,'reason':'candidate_only' if colors else 'color_instability_or_ambiguity','color_diagnostics':evaluations[0][1]}
 return {'status':'candidate' if any(v['colors'] for v in variants.values()) else 'unknown','reason':'version_visual_gate_pending',
  'operating_points':variants,'minimum_jitter_mask_iou':stability,'mask_diagnostics':infos,'model_quality':list(map(float,model_quality)) if model_quality is not None else None,
  'production_accepted':False},geometric[0]
