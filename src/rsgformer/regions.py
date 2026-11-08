"""Directed relation reasoning through overlapping latent regions and geometry feedback."""
import math
import torch
from torch import nn

VARIANTS=('pair','fixed','one_way','feature_membership','geometry_feedback','gnn','shuffled_feedback')

class RegionReasoner(nn.Module):
    def __init__(self,variant='geometry_feedback',hidden=256,projection=128,embedding=64,regions=8,iterations=2):
        super().__init__();assert variant in VARIANTS;assert regions==8
        self.variant=variant;self.hidden=hidden;self.regions=regions;self.iterations=iterations
        self.visual=nn.Sequential(nn.Linear(2048,projection),nn.ReLU())
        self.union=nn.Sequential(nn.Linear(2048,projection),nn.ReLU())
        self.classes=nn.Embedding(151,embedding)
        self.object=nn.Sequential(nn.Linear(projection+embedding+4,hidden),nn.LayerNorm(hidden),nn.ReLU())
        self.pair=nn.Sequential(nn.Linear(3*projection+2*embedding+12,hidden),nn.LayerNorm(hidden),nn.ReLU())
        self.output=nn.Linear(hidden,51)
        self.queries=nn.Parameter(torch.randn(regions,hidden)*.02)
        centers=torch.tensor([[x,y] for y in [.25,.75] for x in [.125,.375,.625,.875]])
        physical=torch.cat([centers,torch.full((regions,2),.6)],-1)
        self.theta=nn.Parameter(torch.logit(physical),requires_grad=variant!='fixed')
        self.query_update=nn.Linear(hidden,hidden)
        self.compatibility=nn.Linear(hidden,hidden,bias=False)
        self.pair_update=nn.GRUCell(hidden,hidden)
        self.context=nn.Sequential(nn.Linear(hidden,hidden),nn.LayerNorm(hidden),nn.ReLU())
        self.predicate_embedding=nn.Embedding(51,embedding)
        self.evidence=nn.Sequential(nn.Linear(hidden+embedding,hidden),nn.LayerNorm(hidden),nn.ReLU())
        self.region_update=nn.GRUCell(hidden,hidden)
        self.geometry_update=nn.Sequential(nn.Linear(2*hidden,hidden),nn.ReLU(),nn.Linear(hidden,4))
        # Both capacity-control and geometry variants use this head in feature updates.
        # Only the application of its four outputs to box logits differs.
        self.delta_to_feature=nn.Linear(4,hidden,bias=False)
        self.gnn_object=nn.Sequential(nn.Linear(hidden,hidden),nn.LayerNorm(hidden),nn.ReLU())
        self.gnn_pair=nn.Linear(2*hidden,hidden)

    @staticmethod
    def decode(theta):
        physical=theta.sigmoid();center=physical[...,:2];extent=physical[...,2:]
        # Constrain extents by available space on both sides; corners stay in [0,1].
        half=extent*torch.minimum(center,1-center)
        return torch.cat([center-half,center+half],-1)

    def membership(self,x,boxes,theta,queries,object_mask):
        regions=self.decode(theta);centers=(boxes[...,:2]+boxes[...,2:])/2
        rc=(regions[...,:2]+regions[...,2:])/2;half=(regions[...,2:]-regions[...,:2])/2
        distance=(centers[:,:,None,:]-rc[:,None,:,:]).abs()/half[:,None,:,:].clamp_min(.02)
        gate=(1-distance).amin(-1)*3
        compatibility=torch.einsum('bnd,bkd->bnk',self.compatibility(x),queries)/math.sqrt(self.hidden)
        alpha=(gate+compatibility).sigmoid()*object_mask[...,None]
        pooled=torch.einsum('bnk,bnd->bkd',alpha,x)/alpha.sum(1)[...,None].clamp_min(1e-5)
        return alpha,pooled,regions

    def forward(self,object_features,union_features,labels,pairs,geometry,boxes,object_mask,pair_mask,frequency_bias=None,diagnostics=False):
        b,n,_=object_features.shape;p=pairs.shape[1];h=self.hidden
        v=self.visual(object_features);c=self.classes(labels)
        x=self.object(torch.cat([v,c,boxes],-1))*object_mask[...,None]
        def gather(t,index):return t.gather(1,index[...,None].expand(-1,-1,t.shape[-1]))
        s,o=pairs[...,0],pairs[...,1]
        z=self.pair(torch.cat([gather(v,s),gather(v,o),gather(c,s),gather(c,o),self.union(union_features),geometry],-1))
        theta=self.theta[None].expand(b,-1,-1);queries=self.queries[None].expand(b,-1,-1)
        alpha,u,region_boxes=self.membership(x,boxes,theta,queries,object_mask)
        initial_boxes=region_boxes;box_history=[region_boxes];alpha_history=[alpha]
        global_context=x.sum(1)/object_mask.sum(1)[:,None].clamp_min(1)
        def logits(value):
            result=self.output(value)
            return result if frequency_bias is None else result+frequency_bias
        if self.variant=='pair':return (logits(z),{}) if diagnostics else logits(z)
        for step in range(self.iterations):
            if self.variant=='gnn':
                messages=self.evidence(torch.cat([z,logits(z).softmax(-1)@self.predicate_embedding.weight],-1))*pair_mask[...,None]
                aggregate=x.new_zeros(b,n,h);degrees=x.new_zeros(b,n,1)
                for indices in [s,o]:
                    aggregate.scatter_add_(1,indices[...,None].expand(-1,-1,h),messages)
                    degrees.scatter_add_(1,indices[...,None],pair_mask[...,None].to(x.dtype))
                x=x+self.gnn_object(aggregate/degrees.clamp_min(1))*object_mask[...,None]
                ctx=self.gnn_pair(torch.cat([gather(x,s),gather(x,o)],-1))
            else:
                support=gather(alpha,s)*gather(alpha,o)*pair_mask[...,None]
                weights=torch.cat([support,support.new_full((b,p,1),.05)],-1)
                weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-5)
                ctx=torch.einsum('bpk,bkd->bpd',weights,torch.cat([u,global_context[:,None]],1))
            z=self.pair_update(self.context(ctx).reshape(-1,h),z.reshape(-1,h)).reshape(b,p,h)
            if self.variant in ['feature_membership','geometry_feedback','shuffled_feedback'] and step<self.iterations-1:
                evidence=self.evidence(torch.cat([z,logits(z).softmax(-1)@self.predicate_embedding.weight],-1))
                if self.variant=='shuffled_feedback':
                    # Roll only within real candidates; padded candidates never provide evidence.
                    evidence=torch.stack([torch.cat([e[:int(m.sum())].roll(1,0),e[int(m.sum()):]],0) for e,m in zip(evidence,pair_mask)])
                support=gather(alpha,s)*gather(alpha,o)*pair_mask[...,None]
                message=torch.einsum('bpk,bpd->bkd',support,evidence)/support.sum(1)[...,None].clamp_min(1e-5)
                updated=self.region_update(message.reshape(-1,h),u.reshape(-1,h)).reshape(b,self.regions,h)
                delta=.1*self.geometry_update(torch.cat([u,message],-1)).tanh()
                updated=updated+self.delta_to_feature(delta)
                if self.variant in ['geometry_feedback','shuffled_feedback']:theta=theta+delta
                # Membership update is common to the capacity control and full method.
                alpha,pooled,region_boxes=self.membership(x,boxes,theta,queries+self.query_update(updated),object_mask)
                u=updated+pooled
                box_history.append(region_boxes);alpha_history.append(alpha)
        result=logits(z)
        if diagnostics:
            return result,{'initial_region_boxes':initial_boxes,'region_boxes':region_boxes,'box_history':box_history,'alpha_history':alpha_history,'alpha':alpha}
        return result
