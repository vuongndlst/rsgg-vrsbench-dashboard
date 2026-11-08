"""Frozen neutral-label object-binding evidence on native crops, not grid panels."""
class NeutralBinding:
 def __init__(self,clip,classes,config):
  self.clip=clip;self.config=config;self.labels=sorted(classes)+config['distractors'];assert len(set(self.labels))==len(self.labels)
  t=clip.torch;prompts=[p.format(object=config['aliases'].get(name,name)) for name in self.labels for p in config['templates']];features=[]
  with t.inference_mode():
   for start in range(0,len(prompts),64):
    f=clip.model.get_text_features(**clip.processor(text=prompts[start:start+64],return_tensors='pt',padding=True).to('cuda'))
    if not isinstance(f,t.Tensor):f=f.pooler_output
    features.append(t.nn.functional.normalize(f.float(),dim=-1))
   self.text=t.nn.functional.normalize(t.cat(features).reshape(len(self.labels),2,-1).mean(1),dim=-1)
 def score(self,images,target):
  import numpy as np
  t=self.clip.torch
  with t.inference_mode():
   f=self.clip.model.get_image_features(**self.clip.processor(images=images,return_tensors='pt').to('cuda'))
   if not isinstance(f,t.Tensor):f=f.pooler_output
   scores=(t.nn.functional.normalize(f.float(),dim=-1)@self.text.T).cpu().numpy()
  family={target}
  for group in self.config['allowed_label_families']:
   if target in group:family.update(group)
  views=[]
  for vector in scores:
   order=np.argsort(-vector);positions={self.labels[int(j)]:rank+1 for rank,j in enumerate(order)}
   best=max(family,key=lambda name:float(vector[self.labels.index(name)]));gap=float(vector[int(order[0])]-vector[self.labels.index(best)])
   views.append({'top5':[{'class':self.labels[int(j)],'cosine':float(vector[j])} for j in order[:5]],'family_member':best,'family_rank':positions[best],'gap_to_best':gap})
  passed={p['id']:all(v['family_rank']<=p['family_rank_max'] and v['gap_to_best']<=p['maximum_gap_to_best'] for v in views) for p in self.config['guard_profiles']}
  return {'native_view_order':['original_bbox','mask_fill127','mask_fill255'],'views':views,'profile_pass':passed,'calibrated_probability':False,'human_verified':False}
