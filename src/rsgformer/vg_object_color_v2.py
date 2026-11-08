"""V2 baseline-independent rectangle GrabCut and explicit 11-color pixel/CLIP candidates; not GT."""
import hashlib,math
import numpy as np
import cv2
from PIL import Image

PALETTE=['black','white','gray','red','green','blue','yellow','brown','orange','pink','purple']
OPERATING_POINTS=[{'pixel_mass':.55,'clip_margin':.01},{'pixel_mass':.65,'clip_margin':.02},{'pixel_mass':.75,'clip_margin':.03}]

def pixel_labels(rgb):
    hsv=cv2.cvtColor(np.asarray(rgb,dtype=np.uint8).reshape(-1,1,3),cv2.COLOR_RGB2HSV).reshape(-1,3)
    h,s,v=hsv.T;labels=np.full(len(h),'gray',dtype='<U6')
    labels[v<45]='black';neutral=(s<45)&(v>=45);labels[neutral&(v>=190)]='white'
    chromatic=(s>=45)&(v>=45)
    for mask,color in [((h<12)|(h>=170),'red'),((h>=12)&(h<25),'orange'),((h>=25)&(h<38),'yellow'),
        ((h>=38)&(h<85),'green'),((h>=85)&(h<130),'blue'),((h>=130)&(h<155),'purple'),((h>=155)&(h<170),'pink')]:
        labels[chromatic&mask]=color
    labels[chromatic&(h<25)&(v<170)]='brown'
    counts={c:int((labels==c).sum()) for c in PALETTE}
    return sorted([(c,n/max(1,len(labels))) for c,n in counts.items()],key=lambda x:(-x[1],PALETTE.index(x[0])))

def prepare_crop(image,box,node_key):
    W,H=image.size;x0,y0,x1,y1=box
    if not all(math.isfinite(x) for x in box) or x1<=x0 or y1<=y0:return {'status':'invalid_bbox'}
    b=[max(0,math.floor(x0)),max(0,math.floor(y0)),min(W,math.ceil(x1)),min(H,math.ceil(y1))]
    if min(b[2]-b[0],b[3]-b[1])<32:return {'status':'small_or_outside_bbox'}
    dx=.15*(b[2]-b[0]);dy=.15*(b[3]-b[1]);p=[max(0,math.floor(b[0]-dx)),max(0,math.floor(b[1]-dy)),min(W,math.ceil(b[2]+dx)),min(H,math.ceil(b[3]+dy))]
    crop=np.asarray(image.crop(p)).copy();scale=min(1,256/max(crop.shape[:2]));crop=cv2.resize(crop,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
    x=int(round((b[0]-p[0])*scale));y=int(round((b[1]-p[1])*scale));w=int(round((b[2]-b[0])*scale));h=int(round((b[3]-b[1])*scale))
    w=min(w,crop.shape[1]-x);h=min(h,crop.shape[0]-y)
    raw=image.crop(b);raw_data={'raw_image':raw,'masked_image':None,'raw_histogram':pixel_labels(np.asarray(raw).reshape(-1,3)),'masked_histogram':[],'clipped_bbox_xyxy':b,'padded_roi_xyxy':p}
    if w*h>=crop.shape[0]*crop.shape[1] or min(w,h)<2:return dict(raw_data,status='no_background_for_grabcut')
    mask=np.zeros(crop.shape[:2],np.uint8);cv2.setRNGSeed(int(hashlib.sha256(node_key.encode()).hexdigest()[:7],16))
    try:cv2.grabCut(cv2.cvtColor(crop,cv2.COLOR_RGB2BGR),mask,(x,y,w,h),np.zeros((1,65)),np.zeros((1,65)),5,cv2.GC_INIT_WITH_RECT)
    except cv2.error:return dict(raw_data,status='grabcut_failure')
    foreground=(mask==cv2.GC_FGD)|(mask==cv2.GC_PR_FGD)
    # Boundary fill never enters pixel statistics.
    count=int(foreground.sum());fraction=count/(w*h)
    if count<256 or not .1<=fraction<=.95:return dict(raw_data,status='mask_quality_failure',foreground_pixels=count,foreground_fraction=fraction)
    masked=np.full_like(crop,127);masked[foreground]=crop[foreground]
    raw=image.crop(b)
    return {'status':'candidate_mask_pass','raw_image':raw,'masked_image':Image.fromarray(masked),
        'raw_histogram':pixel_labels(np.asarray(raw).reshape(-1,3)),
        'masked_histogram':pixel_labels(crop[foreground]),'foreground_pixels':count,
        'foreground_fraction':fraction,'clipped_bbox_xyxy':b,'padded_roi_xyxy':p}

class ClipVerifier:
    def __init__(self,folder):
        import torch
        from transformers import CLIPModel,CLIPProcessor
        self.torch=torch;self.model=CLIPModel.from_pretrained(folder,local_files_only=True).to('cuda').eval()
        self.processor=CLIPProcessor.from_pretrained(folder,local_files_only=True);self.text={}
    def scores(self,images,object_name):
        t=self.torch
        if object_name not in self.text:
            prompts=[s.format(color=c,object=object_name) for c in PALETTE for s in ['a photo of a {color} {object}','a {color} {object}']]+['a photo of a '+object_name]
            inputs=self.processor(text=prompts,return_tensors='pt',padding=True).to('cuda')
            with t.inference_mode():features=self.model.get_text_features(**inputs)
            if not isinstance(features,t.Tensor):features=features.pooler_output
            features=t.nn.functional.normalize(features.float(),dim=-1)
            paired=t.nn.functional.normalize(features[:-1].reshape(len(PALETTE),2,-1).mean(1),dim=-1)
            self.text[object_name]=t.cat([paired,features[-1:]])
        inputs=self.processor(images=images,return_tensors='pt').to('cuda')
        with t.inference_mode():features=self.model.get_image_features(**inputs)
        if not isinstance(features,t.Tensor):features=features.pooler_output
        values=(t.nn.functional.normalize(features.float(),dim=-1)@self.text[object_name].T).cpu().tolist()
        out=[]
        for row in values:
            order=sorted(range(len(PALETTE)),key=lambda i:-row[i]);i,j=order[:2]
            out.append({'top_color':PALETTE[i],'margin':row[i]-row[j],'neutral_object_similarity':row[-1],
                'color_similarities':dict(zip(PALETTE,row[:-1]))})
        return out

def candidates(crop,raw_clip,masked_clip):
    out={}
    for i,point in enumerate(OPERATING_POINTS):
        def pixels(hist):
            if not hist:return []
            a,b=hist[:2]
            if a[1]>=point['pixel_mass']:return [a[0]]
            if a[1]+b[1]>=point['pixel_mass'] and b[1]>=.25:return [a[0],b[0]]
            return []
        def clip(row):return [row['top_color']] if row and row['margin']>=point['clip_margin'] and row['color_similarities'][row['top_color']]>=row['neutral_object_similarity']-.02 else []
        raw=pixels(crop['raw_histogram']);masked=pixels(crop['masked_histogram'])
        agreement=masked if masked_clip and masked_clip['top_color'] in masked and clip(masked_clip) else []
        out[str(i)]={'E0_bbox_pixels':raw,'E1_grabcut_pixels':masked,'E2_clip_bbox':clip(raw_clip),
            'E3_grabcut_clip':clip(masked_clip),'E4_grabcut_pixels_clip_agreement':agreement}
    return out
