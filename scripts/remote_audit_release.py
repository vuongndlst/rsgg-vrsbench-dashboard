"""Full annotation metadata validation, canonical split locks, and satellite feasibility."""
import os,sys,json,hashlib,time,subprocess,zipfile,importlib.metadata
from pathlib import Path
import numpy as np
ROOT=Path.home()/'rsg-former';sys.path[:0]=[str(ROOT/'src'),str(ROOT/'.tools/deps')]
from rsgformer.data import VG150
def digest_ids(ids):return hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()
out={'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'split_reference_commit':'ceb71fa88461c2a97a6258a80f47669d89207296','splits':{},'image_archives_extracted':False}
splits={};reference=None
for split in ['train','val','test']:
    ds=VG150(ROOT/'data/vg150',Path.home()/'sbir/hienlth/vg',split)
    ids=ds.split_image_ids();splits[split]=ids
    info={'images':len(ids),'ordered_image_ids_sha256':ds.split_hash(),'relation_count':0,'object_count':0,'invalid_boxes':0,'self_relations':0,'multi_predicate_pairs':0,'empty_after_clip':0}
    for k in range(len(ds)):
        sample=ds[k];r=sample.relations;info['relation_count']+=len(r);info['object_count']+=len(sample.labels)
        info['invalid_boxes']+=int(((sample.boxes[:,2]<sample.boxes[:,0])|(sample.boxes[:,3]<sample.boxes[:,1])).sum())
        info['self_relations']+=int((r[:,0]==r[:,1]).sum());info['empty_after_clip']+=int(len(r)==0)
        if len(r):
            _,counts=np.unique(r[:,:2],axis=0,return_counts=True);info['multi_predicate_pairs']+=int((counts>1).sum())
    out['splits'][split]=info
    print(json.dumps({'event':'split_metadata_verified','split':split,**info}),flush=True)
    if reference is None:reference=ds
    else:ds.close()
assert not any(set(splits[a])&set(splits[b]) for a,b in [('train','val'),('train','test'),('val','test')])
reference._open_archives();allids=set().union(*map(set,splits.values()));missing=allids-set(reference._member_index)
assert not missing
out['archive_coverage']={'required_split_ids':len(allids),'missing_ids':len(missing),'archive_ids':len(reference._member_index)}
out['h5_schema']=reference.h5_schema;out['h5_split_distribution']={str(int(v)):int(n) for v,n in zip(*np.unique(reference.arrays['split'],return_counts=True))}
out['ordered_reference_metadata_ids_sha256']=digest_ids([i['image_id'] for i in reference.image_info])
out['image_samples']=[{'image_id':reference[k].image_id,'decoded_dimensions':list(reference.image(k).size)} for k in [0,1,2]]
out['environment']={'python':sys.version.split()[0],'packages':{n:importlib.metadata.version(n) for n in ['torch','torchvision','numpy','h5py','Pillow','scipy']},'isolation':'project-owned dependency overlay; inherited torch versions guarded by runtime manifest'}
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
out['resources']={'gpu_snapshot':gpu,'cpu_workers_cap':4,'torch_cpu_threads_cap':4,'p2_gpu_hours_cap':20,'p2_pilot_wall_seconds_cap':1800,'pilot_gpu_memory_gib_cap':8,'p2_generated_artifacts_gib_cap':15,'gpu_reservation':'shared; no exclusive allocation inferred','external_gpu_hour_entitlement':'not_observable; user explicitly authorized bounded P0-P2 execution','storage_available_bytes':os.statvfs(ROOT).f_bavail*os.statvfs(ROOT).f_frsize}
prep=json.loads((ROOT/'reports/P0_PREPARATION.json').read_text());out['star_feasibility']=prep['star']
out['star_feasibility']['image_archive_bytes']=sum(i['bytes'] or 0 for i in prep['star']['files'] if i['name'].startswith('STAR.7z.'))
out['star_feasibility']['status']='annotation_available_full_images_exceed_current_storage'
out['status']='natural_P0_data_ready; evaluator_parity_required_in_P2; satellite_capacity_gate_deferred'
(ROOT/'data/vg150/split_ids.json').write_text(json.dumps(splits,separators=(',',':')))
(ROOT/'reports/P0_RELEASE_AUDIT.json').write_text(json.dumps(out,indent=2))
print(json.dumps({'event':'p0_release_audit_complete','status':out['status'],'report':'reports/P0_RELEASE_AUDIT.json'}),flush=True)
