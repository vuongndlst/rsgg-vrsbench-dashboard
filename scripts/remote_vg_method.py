"""Full VG cache, six controlled screening runs, and gated three-seed confirmation."""
import os,sys,json,time,hashlib,math,random,fcntl,traceback
from pathlib import Path
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
ROOT=Path.home()/'rsg-former';sys.path[:0]=[str(ROOT/'.tools/deps'),str(ROOT/'src')]
import numpy as np,torch
from rsgformer.data import VG150,ImageGraphSample
from rsgformer.frontend import FrozenFrontend
from rsgformer.geometry import directed_pairs
from rsgformer.evaluation import PredClsEvaluator
from rsgformer.regions import RegionReasoner
from rsgformer.vg_experiment import cache_pairs,collate_graphs,sampling_indices,training_targets

CONFIG=json.loads((ROOT/'configs/vg_method_v1.json').read_text())
EXPERIMENT=ROOT/'artifacts'/CONFIG['experiment_id'];REPORT=ROOT/'reports'
DEVICE='cuda';START=time.time();DEADLINE=START+CONFIG['resources']['experiment_wall_hours_max']*3600

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def atomic_json(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp');temporary.write_text(json.dumps(value,indent=2));os.replace(temporary,path)
def event(name,**kwargs):
    record={'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'event':name,**kwargs}
    print(json.dumps(record),flush=True);atomic_json(REPORT/'VG_METHOD_PROGRESS.json',record)
def guard(deadline=None):
    if time.time()>min(DEADLINE,deadline or DEADLINE):raise RuntimeError('wall_budget_exhausted')
    if (EXPERIMENT/'STOP').exists():raise RuntimeError('user_stop_flag')
    st=os.statvfs(ROOT)
    if st.f_bavail*st.f_frsize<CONFIG['resources']['min_storage_free_gib']*1024**3:raise RuntimeError('storage_reserve_reached')
    if torch.cuda.memory_allocated()>CONFIG['resources']['allocator_limit_gib']*1024**3:raise RuntimeError('allocator_cap_reached')
def artifact_guard():
    size=sum(p.stat().st_size for p in (ROOT/'artifacts').rglob('*') if p.is_file())
    if size>CONFIG['resources']['artifact_limit_gib']*1024**3:raise RuntimeError('artifact_cap_reached')
    return size

def initialize():
    global START,DEADLINE
    assert torch.__version__=='2.12.0+cu130' and torch.cuda.is_available()
    tests=json.loads((REPORT/'VG_METHOD_TESTS.json').read_text());assert tests['status']=='PASS'
    assert json.loads((REPORT/'P2_TEST_REPORT.json').read_text())['status']=='PASS'
    EXPERIMENT.mkdir(parents=True,exist_ok=True)
    lock=open(EXPERIMENT/'experiment.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(CONFIG['resources']['cpu_threads']);torch.set_num_interop_threads(2)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    torch.cuda.set_per_process_memory_fraction(CONFIG['resources']['allocator_limit_gib']*1024**3/torch.cuda.get_device_properties(0).total_memory)
    torch.hub.set_dir(str(ROOT/'weights'))
    while torch.cuda.mem_get_info()[0]<(CONFIG['resources']['allocator_limit_gib']+2)*1024**3:
        event('waiting_for_gpu_memory');time.sleep(30)
        if time.time()-START>7200:raise RuntimeError('insufficient_shared_GPU_memory')
    sources=list((ROOT/'src/rsgformer').glob('*.py'))+[Path(__file__)]
    prereg={'config':CONFIG,'config_sha256':digest(ROOT/'configs/vg_method_v1.json'),'sources':{p.relative_to(ROOT).as_posix():digest(p) for p in sources},'started_unix':START,'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'test_scored':False,'device':'cuda'}
    path=EXPERIMENT/'preregistration.json'
    if path.exists():
        previous=json.loads(path.read_text());assert previous['config_sha256']==prereg['config_sha256'] and previous['sources']==prereg['sources'],'immutable_preregistration_changed'
        START=previous['started_unix'];DEADLINE=START+CONFIG['resources']['experiment_wall_hours_max']*3600;prereg=previous
    else:atomic_json(path,prereg)
    atomic_json(REPORT/'VG_METHOD_PREREGISTRATION.json',prereg)
    return lock,prereg

def build_cache():
    cache=EXPERIMENT/'features';cache.mkdir(exist_ok=True)
    lock=json.loads((REPORT/'P0_RELEASE_AUDIT.json').read_text());config_sha=digest(ROOT/'configs/vg_method_v1.json')
    path=cache/'manifest.json';manifest=json.loads(path.read_text()) if path.exists() else {'config_sha256':config_sha,'splits':{},'status':'building'}
    assert manifest['config_sha256']==config_sha
    cache_deadline=START+CONFIG['resources']['screening_wall_hours_max']*3600
    train=VG150(ROOT/'data/vg150',Path.home()/'sbir/hienlth/vg','train');val=VG150(ROOT/'data/vg150',Path.home()/'sbir/hienlth/vg','val')
    assert len(train)==CONFIG['train_images'] and len(val)==CONFIG['validation_images']
    for ds,split in [(train,'train'),(val,'val')]:assert ds.split_hash()==lock['splits'][split]['ordered_image_ids_sha256']
    count_path=EXPERIMENT/'frequency_counts.npy'
    if count_path.exists():counts=np.load(count_path)
    else:
        counts=np.zeros((151,151,51),np.float32)
        for i in range(len(train)):
            s=train[i];r=s.relations
            if len(r):np.add.at(counts,(s.labels[r[:,0]],s.labels[r[:,1]],r[:,2]),1)
            if i%10000==0:guard(cache_deadline)
        np.save(count_path,counts)
    global_counts=counts.sum((0,1));fg_counts=global_counts[1:]
    weights=np.ones(51,np.float32);weights[1:]=np.clip(np.sqrt(np.median(fg_counts)/np.maximum(fg_counts,1)),.25,4)
    np.save(EXPERIMENT/'loss_weights.npy',weights)
    assert int(global_counts.sum())==lock['splits']['train']['relation_count']
    def prior(labels,pairs):
        c=counts[labels[pairs[:,0]],labels[pairs[:,1]]].copy();unseen=c[:,1:].sum(1)==0;c[unseen]=global_counts
        c[:,1:]+=1e-3;c[:,1:]/=np.maximum(c[:,1:].sum(1,keepdims=True),1e-8)
        c[:,1:]=np.log(c[:,1:]);c[:,0]=0
        return torch.from_numpy(c)
    frontend=None
    for split,ds in [('train',train),('val',val)]:
        entries=manifest['splits'].setdefault(split,[])
        for begin in range(0,len(ds),128):
            guard(cache_deadline);shard_number=begin//128;name=f'{split}_{shard_number:05d}.pt';target=cache/name
            if shard_number<len(entries):
                assert target.exists() and target.stat().st_size==entries[shard_number]['bytes'] and digest(target)==entries[shard_number]['sha256'];continue
            if frontend is None:frontend=FrozenFrontend(DEVICE,CONFIG['frontend']['max_side'],CONFIG['frontend']['roi_chunk'],CONFIG['frontend']['stride'])
            rows=[]
            for i in range(begin,min(begin+128,len(ds))):
                guard(cache_deadline);s=ds[i];pairs=cache_pairs(s,CONFIG['train_pair_cache']['negative_candidates'],CONFIG['train_pair_cache']['seed']) if split=='train' else directed_pairs(len(s.labels))
                image=ds.image(i);f=frontend(image,s,pairs)
                scale=torch.tensor([s.width,s.height,s.width,s.height]);f['boxes']=torch.from_numpy(s.boxes.copy())/scale
                f['pixel_boxes']=torch.from_numpy(s.boxes.copy());f['relations']=torch.from_numpy(s.relations.copy());f['global_object_ids']=torch.from_numpy(s.global_object_ids.copy())
                f['width']=s.width;f['height']=s.height;f['frequency_bias']=prior(s.labels,pairs)
                if split=='train':
                    positive=set(map(tuple,s.relations[:,:2]));f['positive']=torch.tensor([tuple(p) in positive for p in pairs],dtype=torch.bool)
                    assert positive<=set(map(tuple,pairs))
                else:assert len(pairs)==len(s.labels)*(len(s.labels)-1)
                rows.append(f)
            temporary=target.with_suffix('.part');torch.save(rows,temporary);os.replace(temporary,target)
            entries.append({'file':name,'bytes':target.stat().st_size,'sha256':digest(target),'image_ids':[r['image_id'] for r in rows]})
            manifest['weights_sha256']=digest(ROOT/'weights/checkpoints/resnet50-11ad3fa6.pth');manifest['counts_sha256']=digest(count_path)
            atomic_json(path,manifest);size=artifact_guard();event('cache_shard_complete',split=split,images_completed=min(begin+128,len(ds)),images_total=len(ds),artifacts_bytes=size,wall_seconds=time.time()-START)
        ds.close()
    del frontend;torch.cuda.empty_cache()
    manifest['status']='complete';atomic_json(path,manifest);event('cache_complete',artifacts_bytes=artifact_guard(),wall_seconds=time.time()-START)
    return manifest

def shards(manifest,split):
    for entry in manifest['splits'][split]:yield torch.load(EXPERIMENT/'features'/entry['file'],weights_only=True,mmap=True,map_location='cpu')
def sample_from_feature(f):
    return ImageGraphSample(f['image_id'],f['width'],f['height'],f['pixel_boxes'].numpy(),f['labels'].numpy(),f['relations'].numpy(),f['global_object_ids'].numpy())

def evaluate(model,manifest,deadline=None,shuffle=False):
    model.eval();e=PredClsEvaluator();records=[];durations=[];diagnostics=[]
    with torch.no_grad():
        for rows in shards(manifest,'val'):
            for begin in range(0,len(rows),CONFIG['optimizer']['batch_images']):
                guard(deadline);batch=rows[begin:begin+CONFIG['optimizer']['batch_images']];args,_=collate_graphs(batch,DEVICE)
                torch.cuda.synchronize();t=time.monotonic();logits,d=model(**args,diagnostics=True);torch.cuda.synchronize();durations.append((time.monotonic()-t)/len(batch));probs=logits.softmax(-1).cpu().numpy()
                if d and model.variant!='gnn':
                    region=d['region_boxes'];areas=(region[...,2:]-region[...,:2]).prod(-1)
                    displacement=(region-d['initial_region_boxes']).abs().mean((1,2))
                    alpha=d['alpha'];occupancy=alpha.sum(1)/args['object_mask'].sum(1)[:,None].clamp_min(1)
                    diagnostics.extend(torch.stack([areas.mean(1),(areas>.8).float().mean(1),displacement,occupancy.mean(1)],1).cpu().tolist())
                for i,f in enumerate(batch):
                    sample=sample_from_feature(f);before=e.images;e.update(sample,f['pairs'].numpy(),probs[i,:len(f['pairs'])])
                    if e.images==before:continue
                    record={'image_id':f['image_id']}
                    for mode in ['gc','ngc']:
                        record[mode+'_R50']=e.recalls[mode][50][-1]
                        recall=np.full(50,np.nan,np.float32)
                        for predicate in np.unique(sample.relations[:,2]):recall[int(predicate)-1]=e.per_pred[mode][50][int(predicate)][-1]
                        record[mode+'_class_R50']=recall
                    records.append(record)
    result=e.result();result['head_ms_per_image_median']=float(np.median(durations)*1000);result['head_ms_per_image_p95']=float(np.percentile(durations,95)*1000)
    result['region_diagnostics']={} if not diagnostics else dict(zip(['mean_region_area','whole_image_fraction','mean_geometry_displacement','mean_membership'],np.mean(diagnostics,0).tolist()))
    return result,records

def save_records(path,records):
    np.savez_compressed(path,image_ids=np.array([r['image_id'] for r in records]),gc_R50=np.array([r['gc_R50'] for r in records]),ngc_R50=np.array([r['ngc_R50'] for r in records]),gc_class_R50=np.stack([r['gc_class_R50'] for r in records]),ngc_class_R50=np.stack([r['ngc_class_R50'] for r in records]))

def frequency_eval(manifest):
    path=REPORT/'VG_FULL_FREQUENCY.json'
    if path.exists():return json.loads(path.read_text())
    e=PredClsEvaluator()
    for rows in shards(manifest,'val'):
        guard()
        for f in rows:
            scores=f['frequency_bias'].exp().numpy();scores[:,0]=0;e.update(sample_from_feature(f),f['pairs'].numpy(),scores)
    result=e.result();result['training_images']=CONFIG['train_images'];result['validation_images']=CONFIG['validation_images'];result['test_scored']=False
    atomic_json(path,result);event('full_frequency_baseline',mR50=result['gc']['mR@50'],R50=result['gc']['R@50']);return result

def checkpoint_interventions(manifest,fullrun):
    checkpoint=torch.load(ROOT/fullrun['checkpoint'],weights_only=True,map_location='cpu')
    m=RegionReasoner('geometry_feedback',**CONFIG['architecture']).to(DEVICE);m.load_state_dict(checkpoint['model'])
    out={'checkpoint':fullrun['checkpoint'],'selection':'diagnostics only; not used for checkpoint or screening gate','test_scored':False,'interventions':{}}
    for variant in ['feature_membership','shuffled_feedback']:
        m.variant=variant;metrics,_=evaluate(m,manifest);out['interventions'][variant]=metrics
        event('checkpoint_intervention',intervention=variant,mR50=metrics['gc']['mR@50'],R50=metrics['gc']['R@50'])
    del m;torch.cuda.empty_cache();atomic_json(REPORT/'VG_FEEDBACK_INTERVENTIONS.json',out)
    return out

def train_run(manifest,phase,variant,seed,epochs):
    run_id=f'{phase}_{variant}_{seed}';folder=EXPERIMENT/run_id;folder.mkdir(exist_ok=True);path=folder/'report.json';last=folder/'last.pt'
    if path.exists():
        result=json.loads(path.read_text())
        if result.get('status')=='complete':return result
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);np.random.seed(seed);random.seed(seed)
    model=RegionReasoner(variant,**CONFIG['architecture']).to(DEVICE)
    opt=torch.optim.AdamW(model.parameters(),lr=CONFIG['optimizer']['learning_rate'],weight_decay=CONFIG['optimizer']['weight_decay'])
    weights=torch.from_numpy(np.load(EXPERIMENT/'loss_weights.npy')).to(DEVICE)
    history=[];best=-math.inf;best_state=None;best_epoch=None;active_count=None;begin_epoch=0;elapsed_previous=0
    if last.exists():
        saved=torch.load(last,weights_only=True,map_location='cpu');assert saved['config_sha256']==digest(ROOT/'configs/vg_method_v1.json')
        model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);history=saved['history'];best=saved['best'];best_state=saved['best_state'];best_epoch=saved['best_epoch'];begin_epoch=saved['epoch'];elapsed_previous=saved['wall_seconds'];active_count=saved['gradient_active_parameters']
    run_start=time.time();run_deadline=run_start+CONFIG['resources']['single_run_wall_hours_max']*3600-elapsed_previous
    if phase=='screening':run_deadline=min(run_deadline,START+CONFIG['resources']['screening_wall_hours_max']*3600)
    for epoch in range(begin_epoch,epochs):
        guard(run_deadline);model.train();rng=np.random.default_rng(np.random.SeedSequence([seed,epoch]));total_loss=0;steps=0;train_images=0;t=time.monotonic()
        multiplier=.1+.9*.5*(1+math.cos(math.pi*epoch/max(epochs-1,1)))
        for group in opt.param_groups:group['lr']=CONFIG['optimizer']['learning_rate']*multiplier
        order=rng.permutation(len(manifest['splits']['train']))
        for order_index,shard_index in enumerate(order):
            rows=torch.load(EXPERIMENT/'features'/manifest['splits']['train'][int(shard_index)]['file'],weights_only=True,mmap=True,map_location='cpu')
            rows=[rows[int(i)] for i in rng.permutation(len(rows))]
            for index in range(0,len(rows),CONFIG['optimizer']['batch_images']):
                guard(run_deadline);batch=rows[index:index+CONFIG['optimizer']['batch_images']]
                indices=[sampling_indices(f,rng,CONFIG['train_sampling']) for f in batch];targets=[training_targets(f,rng) for f in batch]
                args,target=collate_graphs(batch,DEVICE,indices,targets);train_images+=len(batch)
                if not (target!=-100).any():continue
                logits=model(**args);loss=torch.nn.functional.cross_entropy(logits.flatten(0,1),target.flatten(),weight=weights,ignore_index=-100)
                if not torch.isfinite(loss):raise RuntimeError('nonfinite_loss')
                opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),CONFIG['optimizer']['gradient_clip'],error_if_nonfinite=True)
                if active_count is None:active_count=sum(p.numel() for p in model.parameters() if p.grad is not None)
                opt.step();total_loss+=float(loss.detach());steps+=1
            if order_index%40==0:event('training_progress',run=run_id,epoch=epoch+1,shards_done=order_index+1,shards_total=len(order),mean_loss=total_loss/max(steps,1),wall_seconds=time.time()-START)
        assert train_images==CONFIG['train_images']
        metrics,records=evaluate(model,manifest,run_deadline)
        entry={'epoch':epoch+1,'train_loss':total_loss/max(steps,1),'train_images':train_images,'learning_rate':opt.param_groups[0]['lr'],'gc_R50':metrics['gc']['R@50'],'gc_mR50':metrics['gc']['mR@50'],'seconds':time.monotonic()-t,'metrics':metrics};history.append(entry)
        if entry['gc_mR50']>best:
            best=entry['gc_mR50'];best_epoch=epoch+1;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            temporary=folder/'best.part';torch.save({'model':best_state,'variant':variant,'seed':seed,'epoch':best_epoch,'config':CONFIG},temporary);os.replace(temporary,folder/'best.pt');save_records(folder/'best_per_image.npz',records)
        saved={'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},'optimizer':opt.state_dict(),'epoch':epoch+1,'history':history,'best':best,'best_epoch':best_epoch,'best_state':best_state,'config_sha256':digest(ROOT/'configs/vg_method_v1.json'),'wall_seconds':elapsed_previous+time.time()-run_start,'gradient_active_parameters':active_count}
        temporary=folder/'last.part';torch.save(saved,temporary);os.replace(temporary,last);artifact_guard()
        result={'run_id':run_id,'phase':phase,'variant':variant,'seed':seed,'epochs':epochs,'status':'running','history':history,'selected_epoch':best_epoch,'wall_seconds':saved['wall_seconds'],'registered_parameters':sum(p.numel() for p in model.parameters()),'gradient_active_parameters':active_count,'test_scored':False}
        atomic_json(path,result);event('epoch_complete',run=run_id,epoch=epoch+1,loss=entry['train_loss'],R50=entry['gc_R50'],mR50=entry['gc_mR50'],seconds=entry['seconds'])
    best_entry=next(e for e in history if e['epoch']==best_epoch);result['selected_metrics']=best_entry['metrics'];result['status']='complete';result['checkpoint_sha256']=digest(folder/'best.pt');result['checkpoint']=str((folder/'best.pt').relative_to(ROOT));result['per_image']=str((folder/'best_per_image.npz').relative_to(ROOT))
    result['loss_decreased']=history[-1]['train_loss']<history[0]['train_loss'];result['late_mR50_change']=history[-1]['gc_mR50']-history[max(0,len(history)-4)]['gc_mR50'];result['convergence_claim']='fixed epoch budget; inspect curve; convergence not assumed'
    atomic_json(path,result);del model,opt;torch.cuda.empty_cache();event('run_complete',run=run_id,selected_epoch=best_epoch,R50=result['selected_metrics']['gc']['R@50'],mR50=result['selected_metrics']['gc']['mR@50']);return result

def confirmation_statistics(runs):
    seeds=CONFIG['confirmation']['seeds'];full=[r for r in runs if r['variant']=='geometry_feedback'];out={}
    def stats(variant):
        rows=[next(r for r in runs if r['variant']==variant and r['seed']==s) for s in seeds]
        arrays=[np.load(ROOT/r['per_image']) for r in rows]
        return rows,arrays
    fullrows,fullarrays=stats('geometry_feedback');rng=np.random.default_rng(CONFIG['success']['bootstrap_seed'])
    for variant in CONFIG['confirmation']['variants']:
        if variant=='geometry_feedback':continue
        rows,arrays=stats(variant)
        for a,b in zip(fullarrays,arrays):np.testing.assert_array_equal(a['image_ids'],b['image_ids'])
        f=np.nanmean(np.stack([a['gc_class_R50'] for a in fullarrays]),axis=0);c=np.nanmean(np.stack([a['gc_class_R50'] for a in arrays]),axis=0)
        fr=np.mean(np.stack([a['gc_R50'] for a in fullarrays]),axis=0);cr=np.mean(np.stack([a['gc_R50'] for a in arrays]),axis=0)
        diffs=[];rdiffs=[]
        for _ in range(CONFIG['success']['paired_image_bootstrap_replicates']):
            ix=rng.integers(0,len(f),len(f));diffs.append(float((np.nan_to_num(np.nanmean(f[ix],0))-np.nan_to_num(np.nanmean(c[ix],0))).mean()*100));rdiffs.append(float((fr[ix]-cr[ix]).mean()*100))
        delta=np.array([(a['selected_metrics']['gc']['mR@50']-b['selected_metrics']['gc']['mR@50'])*100 for a,b in zip(fullrows,rows)])
        rdelta=np.array([(a['selected_metrics']['gc']['R@50']-b['selected_metrics']['gc']['R@50'])*100 for a,b in zip(fullrows,rows)])
        out[variant]={'delta_mR50_pp_per_seed':delta.tolist(),'mean_delta_mR50_pp':float(delta.mean()),'std_delta_mR50_pp':float(delta.std(ddof=1)),'paired_image_bootstrap_95CI_mR50_pp':np.quantile(diffs,[.025,.975]).tolist(),'mean_delta_R50_pp':float(rdelta.mean()),'paired_image_bootstrap_95CI_R50_pp':np.quantile(rdiffs,[.025,.975]).tolist(),'CI_scope':'image resampling conditional on three trained seeds; not uncertainty over arbitrary training seeds','validation_checkpoint_selection':True}
    return out

def meets_criteria(stats):
    primary=stats['feature_membership'];others=[stats[v] for v in ['pair','one_way','gnn']]
    return primary['mean_delta_mR50_pp']>=CONFIG['success']['delta_mR50_percentage_points'] and primary['mean_delta_R50_pp']>=-CONFIG['success']['max_R50_drop_percentage_points'] and primary['paired_image_bootstrap_95CI_mR50_pp'][0]>0 and all(v['mean_delta_mR50_pp']>0 and v['mean_delta_R50_pp']>=-1 for v in others)

def locked_test(runs):
    ds=VG150(ROOT/'data/vg150',Path.home()/'sbir/hienlth/vg','test');assert len(ds)==CONFIG['locked_test']['images']
    release=json.loads((REPORT/'P0_RELEASE_AUDIT.json').read_text());assert ds.split_hash()==release['splits']['test']['ordered_image_ids_sha256']
    lock={'config_sha256':digest(ROOT/'configs/vg_method_v1.json'),'test_split_hash':ds.split_hash(),'checkpoints':{r['run_id']:r['checkpoint_sha256'] for r in runs},'checkpoint_selection':'validation only; locked before any test prediction','policy':'one pass, all final models, no tuning','locked_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    lockfile=EXPERIMENT/'test_lock.json'
    # Never silently reopen test after a failed/interrupted scoring attempt.
    if lockfile.exists():raise RuntimeError('test_already_opened_do_not_retune_or_restart_silently')
    atomic_json(lockfile,lock);atomic_json(REPORT/'VG_LOCKED_TEST_CONTRACT.json',lock)
    models=[];evaluators=[];records=[[] for _ in runs]
    for run in runs:
        path=ROOT/run['checkpoint'];assert digest(path)==run['checkpoint_sha256']
        m=RegionReasoner(run['variant'],**CONFIG['architecture']).to(DEVICE);m.load_state_dict(torch.load(path,weights_only=True,map_location='cpu')['model']);m.eval();models.append(m);evaluators.append(PredClsEvaluator())
    counts=np.load(EXPERIMENT/'frequency_counts.npy');global_counts=counts.sum((0,1))
    frontend=FrozenFrontend(DEVICE,CONFIG['frontend']['max_side'],CONFIG['frontend']['roi_chunk'],CONFIG['frontend']['stride'])
    with torch.no_grad():
        for begin in range(0,len(ds),CONFIG['optimizer']['batch_images']):
            guard();batch=[]
            for i in range(begin,min(begin+CONFIG['optimizer']['batch_images'],len(ds))):
                s=ds[i];pairs=directed_pairs(len(s.labels));f=frontend(ds.image(i),s,pairs)
                f.update(width=s.width,height=s.height,boxes=torch.from_numpy(s.boxes.copy())/torch.tensor([s.width,s.height,s.width,s.height]),pixel_boxes=torch.from_numpy(s.boxes.copy()),relations=torch.from_numpy(s.relations.copy()),global_object_ids=torch.from_numpy(s.global_object_ids.copy()))
                c=counts[s.labels[pairs[:,0]],s.labels[pairs[:,1]]].copy();c[c[:,1:].sum(1)==0]=global_counts;c[:,1:]+=1e-3;c[:,1:]/=np.maximum(c[:,1:].sum(1,keepdims=True),1e-8);c[:,1:]=np.log(c[:,1:]);c[:,0]=0;f['frequency_bias']=torch.from_numpy(c);batch.append(f)
            args,_=collate_graphs(batch,DEVICE)
            for model_index,(m,e) in enumerate(zip(models,evaluators)):
                probs=m(**args).softmax(-1).cpu().numpy()
                for j,f in enumerate(batch):
                    sample=sample_from_feature(f);before=e.images;e.update(sample,f['pairs'].numpy(),probs[j,:len(f['pairs'])])
                    if e.images==before:continue
                    record={'image_id':f['image_id']}
                    for mode in ['gc','ngc']:
                        record[mode+'_R50']=e.recalls[mode][50][-1];v=np.full(50,np.nan,np.float32)
                        for predicate in np.unique(sample.relations[:,2]):v[int(predicate)-1]=e.per_pred[mode][50][int(predicate)][-1]
                        record[mode+'_class_R50']=v
                    records[model_index].append(record)
            if begin%128==0:event('locked_test_progress',images_completed=min(begin+len(batch),len(ds)),images_total=len(ds),models=len(models))
    test_runs=[]
    for run,e,r in zip(runs,evaluators,records):
        path=EXPERIMENT/(run['run_id']+'_test_per_image.npz');save_records(path,r)
        test_runs.append({'variant':run['variant'],'seed':run['seed'],'selected_metrics':e.result(),'per_image':str(path.relative_to(ROOT)),'checkpoint_sha256':run['checkpoint_sha256'],'test_scored':True})
    del models,frontend;ds.close();torch.cuda.empty_cache();stats=confirmation_statistics(test_runs)
    result={'runs':test_runs,'statistics':stats,'status':'complete','test_scored':True,'lock':lock};atomic_json(REPORT/'VG_LOCKED_TEST_RESULTS.json',result);return result

def main():
    lock,prereg=initialize();event('experiment_started',pid=os.getpid(),config_sha256=prereg['config_sha256'])
    manifest=build_cache();frequency=frequency_eval(manifest);screen=[];screen_deadline=START+CONFIG['resources']['screening_wall_hours_max']*3600
    for variant in CONFIG['screening']['variants']:
        guard(screen_deadline);screen.append(train_run(manifest,'screening',variant,CONFIG['screening']['seed'],CONFIG['screening']['epochs']))
        atomic_json(REPORT/'VG_SCREENING_RESULTS.json',{'runs':screen,'test_scored':False,'frequency':frequency,'status':'running'})
    full=next(r for r in screen if r['variant']=='geometry_feedback')['selected_metrics']['gc'];comparisons={}
    for variant in ['one_way','feature_membership']:
        c=next(r for r in screen if r['variant']==variant)['selected_metrics']['gc'];comparisons[variant]={'delta_mR50_pp':100*(full['mR@50']-c['mR@50']),'delta_R50_pp':100*(full['R@50']-c['R@50'])}
    passed=all(v['delta_mR50_pp']>0 and v['delta_R50_pp']>=-1 for v in comparisons.values())
    atomic_json(REPORT/'VG_SCREENING_RESULTS.json',{'runs':screen,'frequency':frequency,'comparisons':comparisons,'gate_passed':passed,'status':'complete','test_scored':False})
    checkpoint_interventions(manifest,next(r for r in screen if r['variant']=='geometry_feedback'))
    if not passed:
        final={'status':'screening_failed_to_establish_benefit','confirmation_started':False,'screening_comparisons':comparisons,'claim':'insufficient evidence at the preregistered screening budget; no positive result claimed; convergence not assumed','wall_seconds':time.time()-START,'test_scored':False}
        atomic_json(REPORT/'VG_METHOD_RESULT.json',final);event('experiment_finished',**final);return
    confirm=[]
    for seed in CONFIG['confirmation']['seeds']:
        for variant in CONFIG['confirmation']['variants']:
            confirm.append(train_run(manifest,'confirmation',variant,seed,CONFIG['confirmation']['epochs']))
            atomic_json(REPORT/'VG_CONFIRMATION_RESULTS.json',{'runs':confirm,'status':'running','test_scored':False})
    stats=confirmation_statistics(confirm);success=meets_criteria(stats)
    atomic_json(REPORT/'VG_CONFIRMATION_RESULTS.json',{'runs':confirm,'statistics':stats,'status':'complete','test_scored':False,'validation_gate_passed':success})
    final={'status':'validation_criteria_met' if success else 'confirmation_did_not_meet_criteria','confirmation_runs':len(confirm),'statistics':stats,'wall_seconds':time.time()-START,'claim':'controlled validation evidence; no SOTA claim','test_scored':False}
    atomic_json(REPORT/'VG_METHOD_RESULT.json',final)
    if success:
        tested=locked_test(confirm);passed=meets_criteria(tested['statistics']);final.update(status='controlled_test_criteria_met' if passed else 'test_did_not_meet_criteria',test_scored=True,test_statistics=tested['statistics'],wall_seconds=time.time()-START,claim='controlled VG PredCls evidence with the registered frontend/sampling; not published-detector reproduction or SOTA')
    atomic_json(REPORT/'VG_METHOD_RESULT.json',final);event('experiment_finished',**final)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        out={'status':'stopped_on_error','error_type':type(exc).__name__,'reason':str(exc),'test_scored':(EXPERIMENT/'test_lock.json').exists(),'wall_seconds':time.time()-START}
        atomic_json(REPORT/'VG_METHOD_RESULT.json',out);event('experiment_stopped',**out);traceback.print_exc();raise
