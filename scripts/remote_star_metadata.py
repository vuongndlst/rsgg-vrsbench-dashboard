"""Materialize only STAR HDF5/dict/image metadata from the verified annotation ZIP."""
import sys,json,zipfile,hashlib,time
from pathlib import Path
ROOT=Path.home()/'rsg-former';sys.path.insert(0,str(ROOT/'.tools/deps'))
import h5py,numpy as np
folder=ROOT/'data/star';out={'status':'metadata_verified_images_pending_capacity','files':[]}
with zipfile.ZipFile(folder/'STAR_SGG_Annotation.zip') as z:
    for name in ['STAR-SGG-with-attri.h5','STAR-SGG-dicts-with-attri.json','STAR_image_data_v1.json']:
        raw=z.read(name);target=folder/name
        if target.exists():assert hashlib.sha256(target.read_bytes()).digest()==hashlib.sha256(raw).digest()
        else:target.write_bytes(raw)
        out['files'].append({'name':name,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
with h5py.File(folder/'STAR-SGG-with-attri.h5','r') as f:
    out['hdf5_schema']={k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in f.items()}
    out['split_distribution']={str(int(v)):int(n) for v,n in zip(*np.unique(f['split'][:],return_counts=True))}
    out['label_range']=[int(f['labels'][:].min()),int(f['labels'][:].max())];out['predicate_range']=[int(f['predicates'][:].min()),int(f['predicates'][:].max())]
info=json.loads((folder/'STAR-SGG-dicts-with-attri.json').read_text());out['dict_keys']=list(info);out['object_class_count']=len(info.get('idx_to_label',{}));out['predicate_class_count']=len(info.get('idx_to_predicate',{}))
images=json.loads((folder/'STAR_image_data_v1.json').read_text());out['image_metadata_count']=len(images);out['image_metadata_schema']=list(images[0]);out['image_dimension_ranges']={'width':[min(i['width'] for i in images),max(i['width'] for i in images)],'height':[min(i['height'] for i in images),max(i['height'] for i in images)]}
out['geometry_warning']='inspect author OBB/HBB adapter before P6; boxes_1024 shape and coordinate semantics must not be inferred from VG loader'
prep=json.loads((ROOT/'reports/P0_PREPARATION.json').read_text());out['author_release_revision']=prep['star']['revision'];out['declared_license']=prep['star']['card'].get('license');out['image_archive_bytes']=sum(i['bytes'] for i in prep['star']['files'] if i['name'].startswith('STAR.7z.'))
(ROOT/'reports/P0_STAR_METADATA.json').write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)
