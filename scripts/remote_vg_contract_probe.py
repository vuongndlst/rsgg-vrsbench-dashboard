"""Train-only end-to-end timing and optimizer/checkpoint probe before full experiment."""
from pathlib import Path
import os,sys,json,time,hashlib,importlib.util
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
ROOT=Path.home()/'rsg-former';sys.path[:0]=[str(ROOT/'.tools/deps'),str(ROOT/'src')]
import torch,numpy as np
from rsgformer.data import VG150
from rsgformer.frontend import FrozenFrontend
from rsgformer.vg_experiment import cache_pairs,collate_graphs,sampling_indices,training_targets
from rsgformer.regions import RegionReasoner

c=json.loads((ROOT/'configs/vg_method_v1.json').read_text());torch.set_num_threads(4);torch.set_num_interop_threads(2);torch.use_deterministic_algorithms(True)
assert torch.cuda.is_available();torch.cuda.set_per_process_memory_fraction(8*1024**3/torch.cuda.get_device_properties(0).total_memory);torch.hub.set_dir(str(ROOT/'weights'))
ds=VG150(ROOT/'data/vg150',Path.home()/'sbir/hienlth/vg','train');indices=np.random.default_rng(910).choice(len(ds),64,replace=False)
frontend=FrozenFrontend('cuda',c['frontend']['max_side'],128,c['frontend']['stride']);features=[];t=time.monotonic()
counts=np.load(ROOT/'artifacts/train_frequency_counts.npy');global_counts=counts.sum((0,1))
for i in indices:
    s=ds[int(i)];pairs=cache_pairs(s);f=frontend(ds.image(int(i)),s,pairs)
    f.update(width=s.width,height=s.height,boxes=torch.from_numpy(s.boxes.copy())/torch.tensor([s.width,s.height,s.width,s.height]),pixel_boxes=torch.from_numpy(s.boxes.copy()),relations=torch.from_numpy(s.relations.copy()),global_object_ids=torch.from_numpy(s.global_object_ids.copy()))
    p=counts[s.labels[pairs[:,0]],s.labels[pairs[:,1]]].copy();p[p[:,1:].sum(1)==0]=global_counts;p[:,1:]+=1e-3;p[:,1:]/=np.maximum(p[:,1:].sum(1,keepdims=True),1e-8);p[:,1:]=np.log(p[:,1:]);p[:,0]=0;f['frequency_bias']=torch.from_numpy(p)
    positives=set(map(tuple,s.relations[:,:2]));f['positive']=torch.tensor([tuple(p) in positives for p in pairs],dtype=torch.bool);features.append(f)
    if len(features)%16==0:print(json.dumps({'event':'train_probe_features','images':len(features)}),flush=True)
torch.cuda.synchronize();feature_seconds=time.monotonic()-t;del frontend;torch.cuda.empty_cache()
folder=ROOT/'artifacts/vg_method_probe';folder.mkdir(exist_ok=True);torch.save(features,folder/'train64_features.pt');loaded=torch.load(folder/'train64_features.pt',weights_only=True,mmap=True)
out={'status':'PASS','train_images':64,'validation_scored':False,'test_scored':False,'feature_seconds':feature_seconds,'feature_seconds_per_image':feature_seconds/64,'variants':{},'source_checksums':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'src/rsgformer').glob('*.py')}}
for variant in c['screening']['variants']:
    torch.manual_seed(1729);m=RegionReasoner(variant,**c['architecture']).cuda();opt=torch.optim.AdamW(m.parameters(),lr=.0003);rng=np.random.default_rng(1729);losses=[];t=time.monotonic()
    for step in range(12):
        batch=loaded[(step%4)*16:(step%4+1)*16];ix=[sampling_indices(f,rng,c['train_sampling']) for f in batch];targets=[training_targets(f,rng) for f in batch];args,target=collate_graphs(batch,'cuda',ix,targets)
        z=m(**args);loss=torch.nn.functional.cross_entropy(z.flatten(0,1),target.flatten(),ignore_index=-100);assert torch.isfinite(loss)
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5,error_if_nonfinite=True);opt.step();losses.append(float(loss.detach()))
    torch.cuda.synchronize();secs=time.monotonic()-t;m.eval()
    with torch.no_grad():expected=m(**args).detach().cpu()
    torch.save(m.state_dict(),folder/(variant+'.pt'));reload=RegionReasoner(variant,**c['architecture']).cuda();reload.load_state_dict(torch.load(folder/(variant+'.pt'),weights_only=True));reload.eval()
    with torch.no_grad():torch.testing.assert_close(expected,reload(**args).cpu(),rtol=0,atol=0)
    out['variants'][variant]={'seconds_per_batch16':secs/12,'loss_first':losses[0],'loss_last':losses[-1],'checkpoint_reload_exact':True,'parameters':sum(p.numel() for p in m.parameters())};print(json.dumps({'event':'variant_probe_complete','variant':variant,**out['variants'][variant]}),flush=True)
    del m,opt,reload;torch.cuda.empty_cache()
out['peak_torch_allocated_bytes']=torch.cuda.max_memory_allocated();out['projected_cache_wall_hours']=feature_seconds/64*(56224+5000)/3600
(ROOT/'reports/VG_METHOD_CONTRACT_PROBE.json').write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)
