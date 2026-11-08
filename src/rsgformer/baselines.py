import torch
from torch import nn

class PairClassifier(nn.Module):
    def __init__(self,use_union=False,hidden=256,projection=128,embedding=64):
        super().__init__();self.use_union=use_union
        self.visual=nn.Sequential(nn.Linear(2048,projection),nn.ReLU())
        self.classes=nn.Embedding(151,embedding)
        self.union=nn.Sequential(nn.Linear(2048,projection),nn.ReLU()) if use_union else None
        dims=2*projection+2*embedding+12+(projection if use_union else 0)
        self.head=nn.Sequential(nn.Linear(dims,hidden),nn.ReLU(),nn.Linear(hidden,51))
    def forward(self,object_features,union_features,labels,pairs,geometry):
        v=self.visual(object_features);c=self.classes(labels)
        s,o=pairs[:,0],pairs[:,1]
        fields=[v[s],v[o],c[s],c[o],geometry]
        if self.use_union:fields.append(self.union(union_features))
        return self.head(torch.cat(fields,dim=-1))
