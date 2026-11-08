from pathlib import Path
import json,time,os,traceback,random
import numpy as np
import torch
ROOT=Path.home()/'rsg-former'
os.environ['OMP_NUM_THREADS']='4';torch.set_num_threads(4)
torch.cuda.set_per_process_memory_fraction(8*1024**3/torch.cuda.get_device_properties(0).total_memory)
from rsgformer.vg_v2_runtime import native_model,datasets,capture,Adaptation,predicted_graph,SGDetEvaluator
out={'test_scored':False,'validation_scored':False,'status':'RUNNING'}
try:
 model,criterion,config=native_model(ROOT);model.cuda().eval()
 ds=datasets(ROOT,'train');out['train_images']=len(ds);out['train_split_hash']=ds.split_hash()
 out['trainable_native_parameters']=sum(p.numel() for p in model.parameters() if p.requires_grad)
 start=time.time();item,pred=capture(model,ds.image(0));out['frontend_seconds']=time.time()-start
 batch={k:v[None].cuda() for k,v in item.items()}
 native=Adaptation(model,'native').cuda().eval()
 with torch.no_grad():cached,_=native(batch)
 out['cached_fp32_max_abs_error']={k:float((pred[k]-cached[k]).abs().max()) for k in ['pred_logits','pred_boxes','pred_rel','pred_confidence']}
 assert max(out['cached_fp32_max_abs_error'].values())<1e-4,out
 half={k:v.half().float() for k,v in batch.items()}
 with torch.no_grad():quantized,_=native(half)
 out['cached_fp16_max_abs_error']={k:float((pred[k]-quantized[k]).abs().max()) for k in ['pred_rel','pred_confidence']}
 # Choose FP32 caching if half storage measurably perturbs logits.
 out['cache_dtype']='float16' if max(out['cached_fp16_max_abs_error'].values())<.02 else 'float32'
 # Synthetic matching contract: wrong class/localization and empty predictions score zero.
 from rsgformer.data import ImageGraphSample
 s=ImageGraphSample(1,100,100,np.array([[0,0,20,20],[30,30,50,50]],np.float32),np.array([1,2]),np.array([[0,1,3]]),np.array([0,1]))
 ev=SGDetEvaluator();scores=np.zeros((1,50));scores[0,2]=1;ev.update(s,s.boxes,s.labels,np.array([[0,1]]),scores);assert ev.result()['R@50']==1
 ev=SGDetEvaluator();ev.update(s,s.boxes,np.array([1,3]),np.array([[0,1]]),scores);assert ev.result()['R@50']==0
 ev=SGDetEvaluator();ev.update(s,s.boxes,s.labels,np.empty((0,2),int),np.empty((0,50)));assert ev.result()['R@50']==0
 # Compare class/box/predicate matching with the pinned upstream evaluator on
 # synthetic nontrivial predictions. Replace only its legacy bbox ABI function.
 import types,importlib.util
 from rsgformer.geometry import overlaps
 legacy=types.ModuleType('lib.fpn.box_intersections_cpu.bbox');legacy.bbox_overlaps=overlaps
 import sys
 sys.modules['lib.fpn.box_intersections_cpu.bbox']=legacy
 specification=importlib.util.spec_from_file_location('vg_v2_reference_evaluator',ROOT/'vendor/salience_sgg/evaluation/vg_sgg_evaluator.py')
 reference=importlib.util.module_from_spec(specification);specification.loader.exec_module(reference)
 for labels in [s.labels,np.array([1,3])]:
  hits=reference.evaluate_recall(s.relations,s.boxes,s.labels,np.array([[0,1,3]]),s.boxes,labels,np.ones(1),np.ones(2))[0]
  expected=float(len(set().union(*map(set,hits[:50]))))/len(s.relations)
  ev=SGDetEvaluator();ev.update(s,s.boxes,labels,np.array([[0,1]]),scores);assert ev.result()['R@50']==expected
 out['upstream_synthetic_matching_parity']='PASS'
 # Exercise region semantics separately from the zero-initialized residual.
 from rsgformer.vg_v2_runtime import RegionResidual
 rr=RegionResidual('feedback').cuda().eval();x=torch.randn(1,5,256,device='cuda');bx=torch.rand(1,5,4,device='cuda');vis=torch.randn(1,3,256,device='cuda');z=torch.randn(1,5,5,50,device='cuda')
 initial_z,_=rr(x,bx,vis,z);assert torch.equal(initial_z,z),'zero_init_must_preserve_baseline'
 with torch.no_grad():rr.delta.weight.normal_(0,.01)
 learned,m=rr(x,bx,vis,z);learned.square().mean().backward()
 for name in ['tokens','members.weight','feedback.weight']:
  param=dict(rr.named_parameters())[name];assert param.grad is not None and float(param.grad.abs().sum())>0,name
 permutation=torch.tensor([2,0,4,1,3],device='cuda')
 with torch.no_grad():permuted,mp=rr(x[:,permutation],bx[:,permutation],vis,z[:,permutation][:,:,permutation])
 assert torch.allclose(permuted,learned[:,permutation][:,:,permutation],atol=1e-5),'object_ID_permutation_contract'
 out['region_zero_init_feedback_gradient_ID_contracts']='PASS';del rr,initial_z,learned,permuted,mp
 out['variants']={}
 # Target annotations are used only after prediction, by Hungarian loss matching.
 sample=ds[0];boxes=torch.tensor(sample.boxes).cuda();size=boxes.new_tensor([sample.width,sample.height,sample.width,sample.height]);boxes=boxes/size
 target={'boxes':torch.cat([(boxes[:,:2]+boxes[:,2:])/2,boxes[:,2:]-boxes[:,:2]],-1),'labels':torch.tensor(sample.labels-1).cuda(),'rel':torch.zeros(200,200,50,device='cuda')}
 for a,b,p in sample.relations:target['rel'][a,b,p-1]=1
 for variant in ['native','context','oneway','feedback']:
  torch.manual_seed(1729);adapt=Adaptation(model,variant).cuda();adapt.train()
  t=time.time();prediction,member=adapt(batch);losses=criterion(prediction,[target]);loss=sum(losses[k]*criterion.weight_dict[k] for k in losses if k in criterion.weight_dict)
  assert torch.isfinite(loss);loss.backward()
  active=sum(p.numel() for p in adapt.parameters() if p.requires_grad and p.grad is not None)
  out['variants'][variant]={'loss':float(loss),'step_seconds':time.time()-t,'trainable':sum(p.numel() for p in adapt.parameters() if p.requires_grad),'active':active,'peak_bytes':torch.cuda.max_memory_allocated()}
  model.zero_grad(set_to_none=True);del adapt,prediction,losses,loss
  torch.cuda.empty_cache()
 out['status']='PASS'
except Exception as e:
 out.update(status='FAIL',error_type=type(e).__name__,reason=str(e)[:1000],traceback=traceback.format_exc()[-4000:])
t=json.dumps(out,indent=2).replace(str(Path.home()),'${REMOTE_HOME}')
(ROOT/'reports/VG_V2_PROBE.json').write_text(t);print(t)
if out['status']!='PASS':raise SystemExit(1)
