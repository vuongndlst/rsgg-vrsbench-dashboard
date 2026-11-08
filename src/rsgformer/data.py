"""VG150 adapter with canonical ordering and direct reads from existing image ZIPs."""
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
import hashlib,json,zipfile,io
import numpy as np
import h5py
from PIL import Image
from .geometry import overlaps

CORRUPT_IDS={1592,1722,4616,4617}
@dataclass
class ImageGraphSample:
    image_id:int
    width:int
    height:int
    boxes:np.ndarray
    labels:np.ndarray
    relations:np.ndarray
    global_object_ids:np.ndarray
    dataset:str='VG150'
    box_type:str='HBB'
    coordinate_convention:str='original-image inclusive xyxy; normalized from boxes_1024'

class VG150:
    def __init__(self,root,image_root,split='train',num_val=5000,train_overlap=True):
        self.root=Path(root);self.image_root=Path(image_root);self.split=split;self._zips=None
        info=json.loads((self.root/'VG-SGG-dicts-with-attri.json').read_text())
        self.object_names={int(k):v for k,v in info['idx_to_label'].items()};self.predicate_names={int(k):v for k,v in info['idx_to_predicate'].items()}
        assert set(self.object_names)==set(range(1,151)) and set(self.predicate_names)==set(range(1,51))
        self.image_info=[i for i in json.loads((self.root/'image_data.json').read_text()) if i['image_id'] not in CORRUPT_IDS]
        assert len(self.image_info)==108073 and len({i['image_id'] for i in self.image_info})==108073
        with h5py.File(self.root/'VG-SGG-with-attri.h5','r') as f:
            self.h5_schema={k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in f.items()}
            self.arrays={k:f[k][:] for k in ['split','img_to_first_box','img_to_last_box','img_to_first_rel','img_to_last_rel','boxes_1024','labels','relationships','predicates']}
        a=self.arrays;assert len(a['split'])==len(self.image_info)
        assert a['labels'].min()>=1 and a['labels'].max()<=150
        assert a['predicates'].min()>=1 and a['predicates'].max()<=50
        mask=(a['split']==(2 if split=='test' else 0))&(a['img_to_first_box']>=0)&(a['img_to_first_rel']>=0)
        indices=np.where(mask)[0]
        if split=='val':indices=indices[:num_val]
        elif split=='train':indices=indices[num_val:]
        assert split in ['train','val','test']
        self.indices=[]
        for idx in indices:
            if split=='train' and train_overlap:
                sample=self._sample(int(idx),clip=False);r=sample.relations
                b=self._native_boxes(int(idx))
                if not (overlaps(b,b)[r[:,0],r[:,1]]>0).any():continue
            self.indices.append(int(idx))
        self.train_overlap=train_overlap
    def _native_boxes(self,idx):
        a=self.arrays;start=int(a['img_to_first_box'][idx]);end=int(a['img_to_last_box'][idx])+1
        boxes=a['boxes_1024'][start:end].copy()
        assert np.all(boxes[:,2:]>0) and np.all(boxes[:,:2]>=0)
        # Upstream writes floating half-width results back into the HDF5 integer dtype.
        # Preserve its truncation; using float half-pixels changes matching/filtering.
        boxes[:,:2]=boxes[:,:2]-boxes[:,2:]/2
        boxes[:,2:]=boxes[:,:2]+boxes[:,2:]
        return boxes.astype(np.float32)
    def _sample(self,idx,clip=True):
        a=self.arrays;meta=self.image_info[idx];start=int(a['img_to_first_box'][idx]);end=int(a['img_to_last_box'][idx])+1
        boxes=self._native_boxes(idx)
        # Filter training overlap at the native 1024 scale, exactly as upstream (+1 matters).
        r0=int(a['img_to_first_rel'][idx]);r1=int(a['img_to_last_rel'][idx])+1
        relations=np.column_stack([a['relationships'][r0:r1]-start,a['predicates'][r0:r1,0]]).astype(np.int64)
        assert relations[:,:2].min()>=0 and relations[:,:2].max()<len(boxes)
        boxes*=max(meta['width'],meta['height'])/1024
        if clip:
            boxes[:,[0,2]]=np.clip(boxes[:,[0,2]],0,meta['width']-1);boxes[:,[1,3]]=np.clip(boxes[:,[1,3]],0,meta['height']-1)
        return ImageGraphSample(meta['image_id'],meta['width'],meta['height'],boxes,a['labels'][start:end,0].astype(np.int64),relations,np.arange(start,end,dtype=np.int64))
    def __len__(self):return len(self.indices)
    def __getitem__(self,index):
        sample=self._sample(self.indices[index])
        if self.split=='train' and self.train_overlap:
            idx=self.indices[index];b=self._native_boxes(idx)
            r=sample.relations;sample.relations=r[overlaps(b,b)[r[:,0],r[:,1]]>0]
        if self.split=='train':
            valid=(sample.boxes[:,2]>sample.boxes[:,0])&(sample.boxes[:,3]>sample.boxes[:,1])
            remap=np.full(len(valid),-1,dtype=np.int64);remap[valid]=np.arange(valid.sum())
            r=sample.relations;keep=valid[r[:,0]]&valid[r[:,1]]
            sample.relations=np.column_stack([remap[r[keep,:2]],r[keep,2]])
            sample.boxes=sample.boxes[valid];sample.labels=sample.labels[valid];sample.global_object_ids=sample.global_object_ids[valid]
        return sample
    def split_image_ids(self):return [self.image_info[i]['image_id'] for i in self.indices]
    def split_hash(self):return hashlib.sha256(json.dumps(self.split_image_ids(),separators=(',',':')).encode()).hexdigest()
    def _open_archives(self):
        self._zips=[zipfile.ZipFile(self.image_root/name) for name in ['images.zip','images2.zip']]
        self._member_index={}
        for zi,z in enumerate(self._zips):
            for member in z.infolist():
                if member.filename.lower().endswith('.jpg'):
                    try:image_id=int(Path(member.filename).stem)
                    except ValueError:continue
                    assert image_id not in self._member_index
                    self._member_index[image_id]=(zi,member.filename)
    def image(self,index):
        if self._zips is None:self._open_archives()
        meta=self.image_info[self.indices[index]];zi,name=self._member_index[meta['image_id']]
        with self._zips[zi].open(name) as f:
            image=Image.open(io.BytesIO(f.read())).convert('RGB')
        assert image.size==(meta['width'],meta['height'])
        return image
    def close(self):
        if self._zips:
            for z in self._zips:z.close()
        self._zips=None
    def __getstate__(self):
        state=self.__dict__.copy();state['_zips']=None;state.pop('_member_index',None);return state

def ce_targets(sample,pairs,rng):
    """Train-only duplicate sampling; unannotated pairs are benchmark background convention."""
    rels=defaultdict(list)
    for s,o,p in sample.relations:rels[(int(s),int(o))].append(int(p))
    chosen={key:int(rng.choice(labels)) for key,labels in rels.items()}
    return np.array([chosen.get((int(s),int(o)),0) for s,o in pairs],dtype=np.int64)
