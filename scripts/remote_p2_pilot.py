"""Bounded VG150 PredCls pilot: train-only frequency prior and matched A0/A1 heads."""
import os,sys,json,time,hashlib,random,subprocess
from pathlib import Path
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
ROOT=Path.home()/'rsg-former';sys.path[:0]=[str(ROOT/'src'),str(ROOT/'.tools/deps')]
import numpy as np,torch
from rsgformer.data import VG150,ce_targets
from rsgformer.geometry import directed_pairs
from rsgformer.evaluation import PredClsEvaluator
from rsgformer.baselines import PairClassifier
from rsgformer.frontend import FrozenFrontend

config=json.loads((ROOT/'configs/p2_pilot.json').read_text());test=json.loads((ROOT/'reports/P2_TEST_REPORT.json').read_text());assert test['status']=='PASS'
assert torch.__version__=='2.12.0+cu130' and torch.__version__.split('+')[0]=='2.12.0'
torch.set_num_threads(config['torch_cpu_threads']);torch.set_num_interop_threads(2)
torch.manual_seed(config['seed']);np.random.seed(config['seed']);random.seed(config['seed']);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
torch.hub.set_dir(str(ROOT/'weights'))
start=time.monotonic();deadline=start+config['wall_time_limit_seconds']
def say(event,**kw):print(json.dumps({'event':event,**kw}),flush=True)
def guard():
    if time.monotonic()>deadline:raise RuntimeError('P2_pilot_wall_cap_reached')
    generated=sum(p.stat().st_size for p in (ROOT/'artifacts').rglob('*') if p.is_file())
    if generated>config['artifact_limit_gib']*1024**3:raise RuntimeError('P2_artifact_cap_reached')
    if device=='cuda' and torch.cuda.max_memory_allocated()>config['gpu_memory_limit_gib']*1024**3:raise RuntimeError('P2_VRAM_cap_reached')
device='cpu'
if torch.cuda.is_available():
    free=torch.cuda.mem_get_info()[0]
    if free>(config['gpu_memory_limit_gib']+2)*1024**3:
        device='cuda';torch.cuda.set_per_process_memory_fraction(config['gpu_memory_limit_gib']*1024**3/torch.cuda.get_device_properties(0).total_memory)
        torch.cuda.manual_seed_all(config['seed']);torch.cuda.reset_peak_memory_stats()
out={'phase':'P2','config':config,'config_sha256':hashlib.sha256((ROOT/'configs/p2_pilot.json').read_bytes()).hexdigest(),'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'device':device,'claim':'validation engineering smoke; no test scores; not SOTA or official baseline reproduction','runs':{}}
(ROOT/'reports/P2_PILOT_PREREGISTRATION.json').write_text(json.dumps(out,indent=2))
say('pilot_started',device=device,train_images=config['train_images'],validation_images=config['validation_images'])
train=VG150(ROOT/'data/vg150',Path.home()/'sbir/hienlth/vg','train');val=VG150(ROOT/'data/vg150',Path.home()/'sbir/hienlth/vg','val')
out['split_hashes']={'train':train.split_hash(),'val':val.split_hash()};out['pilot_ids']={'train':train.split_image_ids()[:config['train_images']],'val':val.split_image_ids()[:config['validation_images']]}
# Frequency counts only use the train split, retaining all annotated predicates.
counts=np.zeros((151,151,51),np.float32);global_counts=np.zeros(51,np.float32);freq_start=time.monotonic()
for index in range(len(train)):
    sample=train[index];r=sample.relations
    if len(r):
        np.add.at(counts,(sample.labels[r[:,0]],sample.labels[r[:,1]],r[:,2]),1);np.add.at(global_counts,r[:,2],1)
    if index%5000==0:guard()
np.save(ROOT/'artifacts/train_frequency_counts.npy',counts)
def freq_scores(sample,pairs):
    c=counts[sample.labels[pairs[:,0]],sample.labels[pairs[:,1]]].copy();unseen=c[:,1:].sum(1)==0
    c[unseen]=global_counts;c[:,1:]+=1e-3;c[:,0]=0
    c[:,1:]/=np.maximum(c[:,1:].sum(1,keepdims=True),1e-8)
    return c
freq_eval=PredClsEvaluator()
for i in range(config['validation_images']):
    s=val[i];p=directed_pairs(len(s.labels));freq_eval.update(s,p,freq_scores(s,p))
out['runs']['frequency']={'metrics':freq_eval.result(),'train_images_for_prior':len(train),'prior_policy':'foreground object-pair conditional frequencies; train global predicate fallback for unseen pairs; smoothing 1e-3','seconds':time.monotonic()-freq_start}
say('frequency_complete',R50=out['runs']['frequency']['metrics']['gc']['R@50'],mR50=out['runs']['frequency']['metrics']['gc']['mR@50'])

frontend=FrozenFrontend(device,config['backbone_max_side'],config['roi_chunk']);feature_start=time.monotonic();features={}
for split,ds,size in [('train',train,config['train_images']),('val',val,config['validation_images'])]:
    rows=[]
    for i in range(size):
        guard();sample=ds[i];cache=ROOT/'artifacts'/('features_'+split+'_'+str(sample.image_id)+'_'+out['config_sha256'][:12]+'.pt')
        if cache.exists():feat=torch.load(cache,weights_only=True)
        else:
            feat=frontend(ds.image(i),sample);torch.save(feat,cache)
        assert feat['image_id']==sample.image_id and len(feat['pairs'])==len(sample.labels)*(len(sample.labels)-1)
        rows.append(feat)
        if (i+1)%8==0:say('features_ready',split=split,images=i+1)
    features[split]=rows
out['feature_seconds']=time.monotonic()-feature_start
weight_file=ROOT/'weights/checkpoints/resnet50-11ad3fa6.pth'
out['weights']={'source':'https://download.pytorch.org/models/resnet50-11ad3fa6.pth','pretraining':'ImageNet-1K V2','sha256':hashlib.sha256(weight_file.read_bytes()).hexdigest(),'bytes':weight_file.stat().st_size,'frozen':True}
del frontend
if device=='cuda':torch.cuda.empty_cache()

def tensors(feature,indices=None):
    p=feature['pairs'] if indices is None else feature['pairs'][indices]
    u=feature['unions'] if indices is None else feature['unions'][indices]
    g=feature['geometry'] if indices is None else feature['geometry'][indices]
    return (feature['objects'].float().to(device),u.float().to(device),feature['labels'].clone().to(device),p.clone().to(device),g.clone().to(device))
def evaluate(model):
    model.eval();metric=PredClsEvaluator();times=[]
    with torch.no_grad():
        for i,f in enumerate(features['val']):
            guard();s=val[i];scores=[];t=time.monotonic()
            for j in range(0,len(f['pairs']),config['roi_chunk']):
                args=tensors(f,slice(j,j+config['roi_chunk']));scores.append(model(*args).softmax(-1).cpu().numpy())
            probs=np.concatenate(scores) if scores else np.empty((0,51),np.float32)
            if device=='cuda':torch.cuda.synchronize()
            times.append(time.monotonic()-t);metric.update(s,f['pairs'].numpy(),probs)
    return metric.result(),times
for name,use_union in [('A0',False),('A1',True)]:
    guard();torch.manual_seed(config['seed']);rng=np.random.default_rng(config['seed']);model=PairClassifier(use_union,config['hidden_dim'],config['visual_projection_dim'],config['class_embedding_dim']).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay']);run_start=time.monotonic();history=[];best=-1;best_state=None
    for epoch in range(config['epochs']):
        model.train();losses=[]
        for i in rng.permutation(config['train_images']):
            guard();f=features['train'][int(i)];s=train[int(i)];targets=ce_targets(s,f['pairs'].numpy(),rng)
            pos=np.where(targets>0)[0];neg=np.where(targets==0)[0];limit=config['pairs_per_train_image'];n_pos=min(len(pos),int(limit*config['positive_fraction']))
            selected=np.concatenate([rng.choice(pos,n_pos,replace=False) if n_pos else np.array([],dtype=np.int64),rng.choice(neg,min(len(neg),limit-n_pos),replace=False) if len(neg) else np.array([],dtype=np.int64)])
            if not len(selected):continue
            logits=model(*tensors(f,selected));target=torch.from_numpy(targets[selected]).to(device)
            loss=torch.nn.functional.cross_entropy(logits,target);assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);optimizer.step();losses.append(float(loss.detach()))
        metrics,latencies=evaluate(model);entry={'epoch':epoch+1,'loss':float(np.mean(losses)),'R50':metrics['gc']['R@50'],'mR50':metrics['gc']['mR@50']};history.append(entry)
        if entry['mR50']>best:
            best=entry['mR50'];best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};best_metrics=metrics;best_epoch=epoch+1
        say('baseline_epoch',baseline=name,**entry)
    assert best_state is not None
    model.load_state_dict(best_state);checkpoint=ROOT/'artifacts'/(name+'_pilot.pt');torch.save({'model':best_state,'config':config,'baseline':name,'split_hashes':out['split_hashes'],'best_epoch':best_epoch},checkpoint)
    rerun,_=evaluate(model)
    assert rerun['gc']==best_metrics['gc']
    out['runs'][name]={'metrics':best_metrics,'history':history,'selected_epoch':best_epoch,'checkpoint':str(checkpoint.relative_to(ROOT)),'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'parameters':sum(p.numel() for p in model.parameters()),'seconds':time.monotonic()-run_start,'head_latency_ms_median':float(np.median(latencies)*1000),'head_latency_ms_p95':float(np.percentile(latencies,95)*1000),'train_loss_decreased':history[-1]['loss']<history[0]['loss'],'GT_positive_injection_at_evaluation':False}
    (ROOT/'reports/P2_PILOT_REPORT.json').write_text(json.dumps(out,indent=2))
    del model,optimizer
    if device=='cuda':torch.cuda.empty_cache()
out['wall_seconds']=time.monotonic()-start;out['gpu_hours_upper_bound']=out['wall_seconds']/3600 if device=='cuda' else 0
out['peak_gpu_allocated_bytes']=torch.cuda.max_memory_allocated() if device=='cuda' else 0
out['peak_gpu_reserved_bytes']=torch.cuda.max_memory_reserved() if device=='cuda' else 0
out['artifact_bytes']=sum(p.stat().st_size for p in (ROOT/'artifacts').rglob('*') if p.is_file())
out['status']='pilot_complete';out['test_split_scored']=False
(ROOT/'reports/P2_PILOT_REPORT.json').write_text(json.dumps(out,indent=2))
say('p2_pilot_complete',wall_seconds=out['wall_seconds'],peak_vram_gib=out['peak_gpu_allocated_bytes']/1024**3,artifact_gib=out['artifact_bytes']/1024**3,report='reports/P2_PILOT_REPORT.json')
