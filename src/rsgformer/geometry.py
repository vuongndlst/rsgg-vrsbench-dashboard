import numpy as np

def overlaps(a, b):
    """VG toolkit inclusive xyxy IoU convention (+1 pixel), not continuous-box IoU."""
    a=np.asarray(a,dtype=np.float32).reshape(-1,4);b=np.asarray(b,dtype=np.float32).reshape(-1,4)
    wh=np.maximum(np.minimum(a[:,None,2:],b[None,:,2:])-np.maximum(a[:,None,:2],b[None,:,:2])+1,0)
    inter=wh.prod(-1)
    aa=np.maximum(a[:,2:]-a[:,:2]+1,0).prod(-1);bb=np.maximum(b[:,2:]-b[:,:2]+1,0).prod(-1)
    return inter/np.maximum(aa[:,None]+bb[None,:]-inter,1e-8)

def directed_pairs(n):
    i,j=np.where(~np.eye(n,dtype=bool));return np.stack([i,j],axis=1).astype(np.int64)

def union_boxes(boxes,pairs):
    boxes=np.asarray(boxes);pairs=np.asarray(pairs).reshape(-1,2)
    s,o=boxes[pairs[:,0]],boxes[pairs[:,1]]
    return np.concatenate([np.minimum(s[:,:2],o[:,:2]),np.maximum(s[:,2:],o[:,2:])],axis=1)

def pair_geometry(boxes,pairs,width,height):
    s,o=np.asarray(boxes,dtype=np.float32)[np.asarray(pairs)].transpose(1,0,2)
    scale=np.array([width,height,width,height],np.float32)
    sn,on=s/scale,o/scale
    cs=(sn[:,:2]+sn[:,2:])/2;co=(on[:,:2]+on[:,2:])/2
    return np.concatenate([sn,on,co-cs,np.log(np.maximum(on[:,2:]-on[:,:2],1e-6)/np.maximum(sn[:,2:]-sn[:,:2],1e-6))],axis=1).astype(np.float32)
