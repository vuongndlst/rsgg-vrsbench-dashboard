"""Read-only verification of actual first cache shard produced by the full pipeline."""
from pathlib import Path
import json,sys,hashlib
ROOT=Path.home()/'rsg-former';sys.path[:0]=[str(ROOT/'.tools/deps'),str(ROOT/'src')]
import torch,numpy as np
torch.set_num_threads(4)
base=ROOT/'artifacts/vg_method_v1';manifest=json.loads((base/'features/manifest.json').read_text());first=manifest['splits']['train'][0];path=base/'features'/first['file']
assert hashlib.sha256(path.read_bytes()).hexdigest()==first['sha256']
rows=torch.load(path,weights_only=True,mmap=True,map_location='cpu');assert [f['image_id'] for f in rows]==first['image_ids']
for f in rows:
    n=len(f['labels']);p=len(f['pairs']);assert f['objects'].shape==(n,2048) and f['unions'].shape==(p,2048)
    assert f['geometry'].shape==(p,12) and f['frequency_bias'].shape==(p,51)
    for field in ['objects','unions','boxes','geometry','frequency_bias']:assert torch.isfinite(f[field]).all()
    assert n==len(f['global_object_ids']) and len(set(f['global_object_ids'].tolist()))==n
    if n:assert f['labels'].min()>=1 and f['labels'].max()<=150 and f['boxes'].min()>=0 and f['boxes'].max()<=1
    positive=set(map(tuple,f['relations'].numpy()[:,:2]));pairs=list(map(tuple,f['pairs'].numpy()));assert len(set(pairs))==len(pairs)
    assert positive<=set(pairs) and len(set(pairs)-positive)<=64 and all(i!=j for i,j in pairs)
    np.testing.assert_array_equal(f['positive'].numpy(),[p in positive for p in pairs])
prereg=json.loads((ROOT/'reports/VG_METHOD_PREREGISTRATION.json').read_text())
for relative,digest in prereg['sources'].items():assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==digest
out={'status':'PASS','actual_cached_images_checked':len(rows),'source_hashes_match_preregistration':True,'positive_pairs_preserved':True,'stable_object_IDs':True,'invalid_or_duplicate_pairs':0,'test_scored':False}
(ROOT/'reports/VG_METHOD_CACHE_INTEGRITY.json').write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)
