"""Pinned native Salience-SGG bridge, ZIP-backed data and residual region heads.

Upstream inference/loss semantics are kept explicit. The detector cache uses
fixed evaluation transforms: controlled head adaptation, not native recipe reproduction.
"""
from pathlib import Path
import sys,types,importlib,json,inspect,textwrap,random,io,hashlib
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from .data import VG150

def activate_vendor(root):
    vendor=Path(root)/'vendor/salience_sgg'
    sys.path.insert(0,str(Path(root)/'.tools/vg_v2_deps'))
    sys.path.insert(0,str(vendor))
    # Import raw model modules without importing Lightning wrappers/trainers.
    pkg=types.ModuleType('models');pkg.__path__=[str(vendor/'models')];sys.modules['models']=pkg
    ext=types.ModuleType('MultiScaleDeformableAttention');sys.modules['MultiScaleDeformableAttention']=ext
    f=importlib.import_module('models.ops.functions.ms_deform_attn_func')
    def apply(value,shapes,start,locations,weights,step):
        # Use upstream's reference implementation, with autograd, not an old ABI binary.
        return f.ms_deform_attn_core_pytorch(value.float(),shapes,locations.float(),weights.float()).to(value.dtype)
    f.MSDeformAttnFunction.apply=staticmethod(apply)
    # Native backbone constructor normally fetches ImageNet DEFAULT weights even
    # when a complete pretrained VG checkpoint will overwrite every backbone tensor.
    import torchvision
    original=torchvision.models.resnet50
    def resnet50(*args,**kw):kw['weights']=None;return original(*args,**kw)
    torchvision.models.resnet50=resnet50
    return vendor

def native_model(root):
    vendor=activate_vendor(root)
    import yaml
    config=yaml.safe_load((vendor/'configs/SalienceSGG_vg.yaml').read_text())['model']
    for key in ['prior_static','glove_embed']:config[key]=str(vendor/config[key])
    from util.misc import ParserObject
    from models.salience_sgg.deformable_detr import build
    numpy_globals=[(np._core.multiarray._reconstruct,'numpy.core.multiarray._reconstruct'),np.ndarray,np.dtype,np.dtypes.Float32DType,np.dtypes.Float64DType,np.dtypes.Int64DType,np.dtypes.Int32DType]
    with torch.serialization.safe_globals(numpy_globals):
        model,criterion,post=build(ParserObject(config))
    ckpt=torch.load(vendor/'logs/salinece_vg_iou_0.6.ckpt',map_location='cpu',weights_only=False)
    weights=ckpt.get('state_dict',ckpt.get('model',ckpt))
    weights={k[6:] if k.startswith('model.') else k:v for k,v in weights.items() if not k.startswith('criterion.')}
    expected=model.state_dict()
    # Upstream documents this deterministic conversion for older attention
    # checkpoints. Apply only when the target uses separate projection tensors.
    for i in range(6):
        prefix=f'transformer.decoder.layers.{i}.self_attn.'
        if prefix+'q_proj.weight' in expected and prefix+'in_proj_weight' in weights:
            for j,name in enumerate(['q_proj','k_proj','v_proj']):
                weights[prefix+name+'.weight']=weights[prefix+'in_proj_weight'][j*256:(j+1)*256]
                weights[prefix+name+'.bias']=weights[prefix+'in_proj_bias'][j*256:(j+1)*256]
            del weights[prefix+'in_proj_weight'],weights[prefix+'in_proj_bias']
    # Strict loading must establish vocabulary/model shape compatibility.
    model.load_state_dict(weights,strict=True)
    for name,p in model.named_parameters():p.requires_grad_('add_add_add_relation' in name)
    # Extract the unchanged native relation/salience block into a cached-head method.
    from models.salience_sgg import deformable_transformer as module
    code=inspect.getsource(module.DeformableTransformerDecoder.forward)
    first=code.index('            if lid==(self.num_layers-1):')
    last=code.index('            if self.return_intermediate:',first)
    body=textwrap.dedent(code[first:last].split('\n',1)[1])
    # In a singleton ZIP frontend, all valid ratios are1. All final class/box
    # tensors and the final object features are cached, rather than GT inputs.
    prefix='def cached_relation(self, output, boxes, logits):\n    bs=output.shape[0]\n    lid=self.num_layers-1\n    src_valid_ratios=torch.ones(bs,4,2,device=output.device)\n    outputs_classes=[logits]*6\n    outputs_coords=[boxes]*6\n'
    function=prefix+textwrap.indent(body,'    ')+'\n    return relation, torch.stack(confidences)\n'
    ns=dict(module.__dict__);exec(compile(function,'<native_cached_relation>','exec'),ns)
    model.transformer.decoder.cached_relation=types.MethodType(ns['cached_relation'],model.transformer.decoder)
    return model,criterion,config

def datasets(root,split):
    return VG150(Path(root)/'data/vg150',Path.home()/'sbir/hienlth/vg',split=split,train_overlap=False)

def image_tensor(image):
    import torchvision.transforms.functional as TF
    w,h=image.size;scale=min(800/min(w,h),1333/max(w,h))
    image=image.resize((int(round(w*scale)),int(round(h*scale))))
    return TF.normalize(TF.to_tensor(image),[.485,.456,.406],[.229,.224,.225])

def capture(model,image):
    grabbed={}
    def decoder_hook(m,args,output):grabbed['nodes']=output[0][-1].detach()
    def proj_hook(m,args,output):
        level=len(grabbed.setdefault('visual',[]))
        pooled=F.adaptive_avg_pool2d(output,(4,4) if level==0 else (1,1))
        grabbed['visual'].append(pooled.flatten(2).transpose(1,2).detach())
    hooks=[model.transformer.register_forward_hook(decoder_hook)]
    hooks += [p.register_forward_hook(proj_hook) for p in model.input_proj]
    try:
        with torch.no_grad():out=model([image_tensor(image).cuda()])
    finally:
        for hook in hooks:hook.remove()
    item={'nodes':grabbed['nodes'][0].cpu(),'boxes':out['pred_boxes'][0].cpu(),'logits':out['pred_logits'][0].cpu(),'visual':torch.cat(grabbed['visual'],1)[0].cpu()}
    return item,out

class RegionResidual(nn.Module):
    """Matched active parameters across context/one-way/feedback residual controls."""
    def __init__(self,variant,dim=256,regions=8,rounds=2):
        super().__init__();self.variant=variant;self.rounds=rounds
        self.tokens=nn.Parameter(torch.randn(regions,dim)*.02)
        self.box=nn.Linear(4,dim);self.att=nn.MultiheadAttention(dim,8,batch_first=True)
        self.norm=nn.LayerNorm(dim);self.members=nn.Linear(dim,dim)
        self.pair=nn.Sequential(nn.Linear(dim*3+50,dim),nn.GELU(),nn.Linear(dim,dim))
        self.feedback=nn.Linear(dim,dim);self.delta=nn.Linear(dim,50);self.gate=nn.Parameter(torch.tensor(-3.))
        nn.init.zeros_(self.delta.weight);nn.init.zeros_(self.delta.bias)
    def forward(self,nodes,boxes,visual,base):
        b,n,d=nodes.shape;x=nodes+self.box(boxes);regions=self.tokens[None].expand(b,-1,-1)
        regions=self.norm(regions+self.att(regions,torch.cat([x,visual],1),torch.cat([x,visual],1),need_weights=False)[0])
        membership=None
        # All variants use the same projections/pair feedback operations. The
        # context control assigns every pair the same token-average context.
        for iteration in range(self.rounds):
            raw=torch.einsum('bnd,bkd->bnk',self.members(x),regions)/d**.5
            membership=raw.sigmoid()
            pair_weights=membership[:,:,None,:]*membership[:,None,:,:]
            local=torch.einsum('bijk,bkd->bijd',pair_weights,regions)/(pair_weights.sum(-1,keepdim=True)+1e-6)
            global_context=regions.mean(1)[:,None,None,:].expand(-1,n,n,-1)
            context=global_context if self.variant=='context' else .75*local+.25*global_context
            pair=self.pair(torch.cat([x[:,:,None,:].expand(-1,-1,n,-1),x[:,None,:,:].expand(-1,n,-1,-1),context,base.sigmoid()],-1))
            weights=pair_weights/(pair_weights.sum((1,2),keepdim=True)+1e-6)
            message=torch.einsum('bijk,bijd->bkd',weights,pair)
            feedback=self.feedback(message)
            if self.variant=='feedback':regions=self.norm(regions+.1*feedback)
            else:
                # Keep feedback parameters active with identical context sources;
                # do not alter region memberships in the one-way/context controls.
                pair=pair+.1*feedback.mean(1)[:,None,None,:]
        return base+self.gate.sigmoid()*self.delta(pair),membership

class Adaptation(nn.Module):
    def __init__(self,model,variant):
        super().__init__();self.native=model;self.variant=variant
        self.residual=None if variant=='native' else RegionResidual(variant)
    def forward(self,batch):
        base,confidence=self.native.transformer.decoder.cached_relation(batch['nodes'],batch['boxes'],batch['logits'])
        membership=None
        if self.residual is not None:base,membership=self.residual(batch['nodes'],batch['boxes'],batch['visual'],base)
        return {'pred_logits':batch['logits'],'pred_boxes':batch['boxes'],'pred_rel':base,'pred_confidence':confidence},membership

class SGDetEvaluator:
    def __init__(self):
        self.r={k:[] for k in [20,50,100]};self.pr={k:[[] for _ in range(50)] for k in self.r};self.records=[]
    def update(self,sample,boxes,classes,pairs,scores):
        from .geometry import overlaps
        gt=sample.relations;classes=np.asarray(classes);pred=1+scores.argmax(1)
        order=np.argsort(-scores.max(1),kind='stable')[:100];pairs=pairs[order];pred=pred[order]
        # Scores already include both object and native salience confidence.
        iou=overlaps(sample.boxes,np.asarray(boxes));hit=[]
        for (s,o),p in zip(pairs,pred):
            mask=(gt[:,2]==p)&(sample.labels[gt[:,0]]==classes[s])&(sample.labels[gt[:,1]]==classes[o])&(iou[gt[:,0],s]>=.5)&(iou[gt[:,1],o]>=.5)
            hit.append(set(np.where(mask)[0]))
        record={'image_id':sample.image_id,'recalls':{},'predicates':{}}
        for k in self.r:
            matched=set().union(*hit[:k]) if hit else set();rec=len(matched)/max(1,len(gt));self.r[k].append(rec);record['recalls'][str(k)]=rec
            for p in np.unique(gt[:,2]):
                inds=set(np.where(gt[:,2]==p)[0]);v=len(inds&matched)/len(inds);self.pr[k][p-1].append(v)
                record['predicates'].setdefault(str(p),{})[str(k)]=v
        self.records.append(record)
    def result(self):
        o={'images':len(self.records),'mode':'SGDet','constraint':'GC','matching':'class AND subject/object inclusive-IoU>=0.5'}
        for k in self.r:
            r=float(np.mean(self.r[k]));mr=float(np.mean([np.mean(x) if x else 0 for x in self.pr[k]]))
            o.update({f'R@{k}':r,f'mR@{k}':mr,f'F@{k}':2*r*mr/max(1e-12,r+mr)})
        return o

def predicted_graph(out,sample):
    """Native VG postprocessing: softmax(sigmoid object logits), NMS merge,
    max-predicate GC with object scores and final salience confidence.
    """
    import torchvision
    from util.box_ops import box_cxcywh_to_xyxy
    logits=out['pred_logits'][0].sigmoid().softmax(-1);objscore,classes=logits.max(-1)
    boxes=box_cxcywh_to_xyxy(out['pred_boxes'][0]);n=len(boxes)
    kept=torchvision.ops.batched_nms(boxes,objscore,classes,.5)
    ious=torchvision.ops.box_iou(boxes,boxes[kept]);assigned=torch.zeros_like(classes)
    for c in torch.unique(classes[kept]):
        a=torch.where(classes==c)[0];b=torch.where(classes[kept]==c)[0]
        assigned[a]=kept[b[ious[a][:,b].argmax(1)]]
    rel=out['pred_rel'][0].sigmoid();conf=out['pred_confidence'][-1,0].sigmoid()
    score=rel*conf[:,:,None]*objscore[:,None,None]*objscore[None,:,None]
    score[torch.arange(n),torch.arange(n)]=0
    order=torch.argsort(score.max(-1).values.flatten(),descending=True)
    pairs=torch.stack([order//n,order%n],-1);merged=assigned[pairs]
    # One prediction per merged object pair, as the official NMS assignment.
    seen=set();inds=[]
    predicates=score[pairs[:,0],pairs[:,1]].argmax(-1).cpu().tolist()
    for i,p in enumerate(merged.cpu().tolist()):
        key=tuple(p)+(predicates[i],)
        if key not in seen:seen.add(key);inds.append(i)
        if len(inds)>=100:break
    pairs=pairs[inds];scores=score[pairs[:,0],pairs[:,1]]
    scale=boxes.new_tensor([sample.width,sample.height,sample.width,sample.height]);boxes=boxes*scale
    return boxes.cpu().numpy(),classes.cpu().numpy()+1,pairs.cpu().numpy(),scores.cpu().numpy()
