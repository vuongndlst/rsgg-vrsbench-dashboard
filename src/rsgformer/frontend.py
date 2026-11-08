"""Frozen ImageNet frontend for controlled baselines; not the official VG detector."""
import torch
from torch import nn
from torchvision.models import resnet50,ResNet50_Weights
from torchvision.ops import roi_align
from torchvision.transforms.functional import pil_to_tensor
from .geometry import directed_pairs,union_boxes,pair_geometry

class FrozenFrontend:
    def __init__(self,device='cpu',max_side=384,chunk=128,stride=32):
        assert stride in (16,32)
        self.device=torch.device(device);self.max_side=max_side;self.chunk=chunk;self.stride=stride
        model=resnet50(weights=ResNet50_Weights.IMAGENET1K_V2,replace_stride_with_dilation=[False,False,stride==16])
        self.encoder=nn.Sequential(*list(model.children())[:-2]).eval().to(self.device)
        for p in self.encoder.parameters():p.requires_grad_(False)
    @torch.inference_mode()
    def __call__(self,image,sample,pairs=None):
        w,h=image.size;scale=self.max_side/max(w,h)
        nw=max(32,round(w*scale/32)*32);nh=max(32,round(h*scale/32)*32)
        im=image.resize((nw,nh))
        x=pil_to_tensor(im).float().div(255).to(self.device)
        x=(x-torch.tensor([.485,.456,.406],device=self.device)[:,None,None])/torch.tensor([.229,.224,.225],device=self.device)[:,None,None]
        maps=self.encoder(x[None]);pairs=directed_pairs(len(sample.labels)) if pairs is None else pairs;unions=union_boxes(sample.boxes,pairs)
        def pool(boxes):
            b=torch.as_tensor(boxes,dtype=torch.float32,device=self.device).clone()
            b[:,2:]+=1 # inclusive xyxy -> half-open endpoints for visual pooling
            b[:,[0,2]]*=nw/w;b[:,[1,3]]*=nh/h
            chunks=[roi_align(maps,[b[i:i+self.chunk]],output_size=1,spatial_scale=1/self.stride,aligned=False).flatten(1).cpu() for i in range(0,len(b),self.chunk)]
            return torch.cat(chunks).half() if chunks else torch.empty((0,2048),dtype=torch.float16)
        return {'image_id':sample.image_id,'objects':pool(sample.boxes),'unions':pool(unions),'labels':torch.as_tensor(sample.labels),'pairs':torch.as_tensor(pairs),'geometry':torch.from_numpy(pair_geometry(sample.boxes,pairs,w,h))}
