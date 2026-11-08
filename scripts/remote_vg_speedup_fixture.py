"""CPU-only contracts for confidence-axis slicing and per-image loss weighting."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import json, signal, hashlib
from pathlib import Path
signal.alarm(45)
import torch
torch.set_num_threads(1)
from rsgformer.vg_speedup import singleton,per_image_losses,finite_loss,pack
torch.manual_seed(71)
pred={'pred_rel':torch.randn(4,3,3,2,requires_grad=True),
      'pred_confidence':torch.randn(2,4,3,3,requires_grad=True),
      'pred_logits':torch.randn(4,3,5),'pred_boxes':torch.rand(4,3,4)}
class Criterion:
    weight_dict={'r':2.,'c':3.}
    def __call__(self,out,targets):
        assert len(targets)==1 and out['pred_confidence'].shape==(2,1,3,3)
        factor=targets[0]['factor']
        return {'r':out['pred_rel'].square().sum()/factor,
                'c':out['pred_confidence'].square().sum()/factor}
c=Criterion();targets=[{'factor':x} for x in [1,2,5,11]]
loss=per_image_losses(c,pred,targets)
reference=torch.stack([sum(v*c.weight_dict[k] for k,v in c(singleton(pred,i),[targets[i]]).items()) for i in range(4)])
assert torch.equal(loss,reference)
g1=torch.autograd.grad(loss.sum()/16,[pred['pred_rel'],pred['pred_confidence']],retain_graph=True)
g2=torch.autograd.grad(sum(reference)/16,[pred['pred_rel'],pred['pred_confidence']])
assert all(torch.equal(a,b) for a,b in zip(g1,g2))
for i in range(4):
    actual=singleton(pred,i)
    assert torch.equal(actual['pred_confidence'][:,0],pred['pred_confidence'][:,i])
    assert torch.equal(actual['pred_rel'][0],pred['pred_rel'][i])
for x in [float('nan'),float('inf')]:
    rejected=False
    try:finite_loss(torch.tensor([1.,x]))
    except RuntimeError:rejected=True
    assert rejected
items=[{k:torch.full((2,3),float(i)) for k in ('nodes','boxes','logits','visual')} for i in range(4)]
packed=pack(items,device='cpu')
assert all(torch.equal(packed[k][i],items[i][k]) for k in packed for i in range(4))
root=Path.home()/'rsg-former'
report={'status':'PASS','tests':['confidence_layer_batch_axis','per_image_normalization','gradient_sum_effective_batch16','nan_inf_rejection','ordered_fp32_collation'],'gpu_used':False,'real_dataset_accessed':False,
        'module_sha256':hashlib.sha256((root/'src/rsgformer/vg_speedup.py').read_bytes()).hexdigest()}
(root/'reports/VG_SPEEDUP_FIXTURE.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report))
