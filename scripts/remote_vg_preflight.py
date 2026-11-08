"""Read-only feasibility for full VG training and full validation; no test scoring."""
from pathlib import Path
import json,sys,os,time,numpy as np
ROOT=Path.home()/'rsg-former';sys.path[:0]=[str(ROOT/'.tools/deps'),str(ROOT/'src')]
from rsgformer.data import VG150
import torch,torchvision
torch.set_num_threads(4)
out={'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'splits':{},'environment':{'torch':torch.__version__,'torchvision':torchvision.__version__},'test_scored':False}
for split in ['train','val']:
    ds=VG150(ROOT/'data/vg150',Path.home()/'sbir/hienlth/vg',split);objects=pairs=cache_pairs=0;max_objects=0
    for i in range(len(ds)):
        s=ds[i];n=len(s.labels);npairs=n*(n-1);positive=len(np.unique(s.relations[:,:2],axis=0)) if len(s.relations) else 0
        objects+=n;pairs+=npairs;cache_pairs+=positive+min(64,npairs-positive) if split=='train' else npairs;max_objects=max(max_objects,n)
    out['splits'][split]={'images':len(ds),'objects':objects,'all_directed_pairs':pairs,'cached_pairs':cache_pairs,'max_objects':max_objects,'estimated_fp16_features_bytes':(objects+cache_pairs)*2048*2,'split_hash':ds.split_hash()};ds.close()
out['storage_available_bytes']=os.statvfs(ROOT).f_bavail*os.statvfs(ROOT).f_frsize
if torch.cuda.is_available():
    out['gpu_free_bytes'],out['gpu_total_bytes']=torch.cuda.mem_get_info()
out['projected_feature_bytes']=sum(v['estimated_fp16_features_bytes'] for v in out['splits'].values())
(ROOT/'reports/VG_METHOD_PREFLIGHT.json').write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)
