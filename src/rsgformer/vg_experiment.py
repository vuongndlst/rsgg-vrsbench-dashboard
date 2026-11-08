"""Deterministic train-only sampling and batched padded graphs for full VG experiments."""
import numpy as np
import torch
from .geometry import directed_pairs
from .data import ce_targets

def cache_pairs(sample,negative_candidates=64,seed=9181):
    allpairs=directed_pairs(len(sample.labels))
    positive=set(map(tuple,sample.relations[:,:2]))
    keep=np.array([tuple(p) in positive for p in allpairs],dtype=bool)
    indices=np.where(~keep)[0];rng=np.random.default_rng(np.random.SeedSequence([seed,sample.image_id]))
    if len(indices)>negative_candidates:indices=rng.choice(indices,negative_candidates,replace=False)
    selected=np.sort(np.concatenate([np.where(keep)[0],indices]))
    return allpairs[selected]

def sampling_indices(feature,rng,policy):
    positives=np.where(feature['positive'].numpy())[0];negatives=np.where(~feature['positive'].numpy())[0]
    npick=min(len(positives),policy['max_positive_pairs'])
    nk=min(len(negatives),npick*policy['negative_to_positive_ratio'] if npick else policy['empty_positive_negative_pairs'])
    return np.concatenate([rng.choice(positives,npick,replace=False),rng.choice(negatives,nk,replace=False)])

def collate_graphs(features,device='cpu',indices=None,targets=None):
    b=len(features);n=max(1,max(len(f['labels']) for f in features));p=max(1,max(len(f['pairs']) if indices is None else len(indices[i]) for i,f in enumerate(features)))
    out={'object_features':torch.zeros(b,n,2048),'union_features':torch.zeros(b,p,2048),'labels':torch.zeros(b,n,dtype=torch.long),'pairs':torch.zeros(b,p,2,dtype=torch.long),'geometry':torch.zeros(b,p,12),'boxes':torch.zeros(b,n,4),'object_mask':torch.zeros(b,n,dtype=torch.bool),'pair_mask':torch.zeros(b,p,dtype=torch.bool),'frequency_bias':torch.zeros(b,p,51)}
    target=torch.full((b,p),-100,dtype=torch.long)
    for i,f in enumerate(features):
        k=len(f['labels']);ix=slice(None) if indices is None else indices[i];q=len(f['pairs'][ix])
        out['object_features'][i,:k]=f['objects'];out['labels'][i,:k]=f['labels'];out['boxes'][i,:k]=f['boxes'];out['object_mask'][i,:k]=True
        for key,source in [('union_features','unions'),('pairs','pairs'),('geometry','geometry'),('frequency_bias','frequency_bias')]:out[key][i,:q]=f[source][ix]
        out['pair_mask'][i,:q]=True
        if targets is not None:target[i,:q]=targets[i][ix]
    return {k:v.to(device) for k,v in out.items()},target.to(device)

def training_targets(feature,rng):
    # Reconstruct only GT annotations; padded/held-out labels never enter this function.
    from .data import ImageGraphSample
    sample=ImageGraphSample(feature['image_id'],feature['width'],feature['height'],feature['pixel_boxes'].numpy(),feature['labels'].numpy(),feature['relations'].numpy(),feature['global_object_ids'].numpy())
    return torch.from_numpy(ce_targets(sample,feature['pairs'].numpy(),rng))
