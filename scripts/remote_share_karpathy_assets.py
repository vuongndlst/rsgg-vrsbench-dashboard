"""Move existing Karpathy metadata/images into shared-assets, preserving old paths.

No downloads, generation, evaluation, training or VG asset modifications.
Exact metadata SHA and test membership determine completeness.
"""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import os,json,re,hashlib,ctypes
home=Path.home().resolve();shared=home/'shared-assets';events=[]
def alias(x):return re.sub(r'/(home|users)/[^/\s]+',r'/\1/${ACCOUNT}',str(x).replace(str(home),'${REMOTE_HOME}').replace(home.name,'${ACCOUNT}'))
def boundary(p):
    assert p!=home and p.is_relative_to(home) and p.parent.resolve().is_relative_to(home)
    if not p.is_symlink():assert p.resolve().is_relative_to(home)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
libc=ctypes.CDLL(None,use_errno=True);exchange=libc.renameat2
exchange.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint];exchange.restype=ctypes.c_int
def log(e):
    events.append(e)
    (shared/'maintenance/KARPATHY_MOVE_20261010.json').write_text(json.dumps(events,indent=2))
def move(src,dst):
    boundary(src);boundary(dst)
    if src.is_symlink() and src.resolve()==dst and dst.exists():
        log({'action':'already_shared','source':alias(src),'destination':alias(dst)});return
    assert src.exists() and not src.is_symlink() and not os.path.lexists(dst)
    assert src.stat().st_uid==os.getuid()
    dst.parent.mkdir(parents=True,exist_ok=True)
    before=(src.stat().st_dev,src.stat().st_ino);assert before[0]==dst.parent.stat().st_dev
    dst.symlink_to(dst,target_is_directory=src.is_dir())
    log({'action':'move_prepared','source':alias(src),'destination':alias(dst)})
    if exchange(-100,os.fsencode(src),-100,os.fsencode(dst),2):
        err=ctypes.get_errno();dst.unlink();raise OSError(err,'atomic_exchange_failed')
    assert src.resolve()==dst and before==(dst.stat().st_dev,dst.stat().st_ino)
    log({'action':'moved','source':alias(src),'destination':alias(dst),'same_inode':True})
metadata=[('coco_karpathy','skgcap_vlm/data/dataset_coco.json','2fd999220673258012acfb411a4e7e66af7d488050b2519b0badcc49b7600b8d'),('flickr30k_karpathy','skgcap_vlm/data/dataset_flickr30k.json','db779ee2c5df40c25ff0eebeca464f6124b620cc4a9398650aea71fd1b53d4af')]
for name,rel,expected in metadata:
    src=home/rel;dst=shared/'datasets'/name/'annotations'/src.name
    assert sha(src)==expected
    move(src,dst);assert sha(dst)==expected
move(home/'skgcap_vlm/images',shared/'datasets/coco_karpathy/images')
move(home/'image-captioning-kg/datasets/flickr30k/images',shared/'datasets/flickr30k_karpathy/images')
# Preserve consumers of the byte-identical COCO metadata while consolidating it.
duplicate=home/'coco_kb_v3/data/dataset_coco.json';canonical=shared/'datasets/coco_karpathy/annotations/dataset_coco.json'
if not duplicate.is_symlink():
    boundary(duplicate);assert sha(duplicate)==sha(canonical)
    log({'action':'metadata_duplicate_prepared','path':alias(duplicate),'sha256':sha(canonical)})
    duplicate.unlink();duplicate.symlink_to(canonical)
    log({'action':'metadata_duplicate_replaced_with_alias','path':alias(duplicate)})
datasets={}
for name,rel,expected in metadata:
    root=shared/'datasets'/name;data=json.loads((root/'annotations'/Path(rel).name).read_text());test=[i for i in data['images'] if i['split']=='test']
    records=[];missing=[];annotations=[];ann_id=1
    for item in test:
        path=root/'images'/item['filename'];present=path.is_file() and path.stat().st_size>0
        if not present:missing.append(item['filename'])
        image_id=int(item.get('cocoid',item['imgid']))
        records.append({'id':image_id,'karpathy_imgid':item['imgid'],'file_name':item['filename'],'image_path':alias(path),'image_present':present,'file_bytes':path.stat().st_size if present else None})
        for sent in item['sentences']:
            annotations.append({'id':ann_id,'image_id':image_id,'caption':sent['raw']});ann_id+=1
    assert len({r['id'] for r in records})==len(records)
    for target,content in [('test_manifest.json',{'dataset':name,'split':'Karpathy test','metadata_sha256':expected,'images':records}),('test_references_coco.json',{'info':{'description':name+' Karpathy test human captions'},'licenses':[],'images':[{'id':r['id'],'file_name':r['file_name']} for r in records],'annotations':annotations}),('missing_test_images.json',{'dataset':name,'missing_count':len(missing),'filenames':missing})]:
        (root/target).write_text(json.dumps(content,ensure_ascii=False))
    datasets[name]={'root':alias(root),'metadata_sha256':expected,'split_counts':dict(Counter(i['split'] for i in data['images'])),'expected_test_images':len(test),'available_test_images':len(test)-len(missing),'missing_test_images':len(missing),'human_reference_captions':len(annotations),'status':'TEST_IMAGE_COVERAGE_COMPLETE' if not missing else 'INCOMPLETE_DO_NOT_SCORE_AS_FULL_TEST','image_files_moved_total':sum(p.is_file() for p in (root/'images').iterdir()),'test_manifest_sha256':sha(root/'test_manifest.json'),'references_sha256':sha(root/'test_references_coco.json')}
qwen=home/'etechs/msvd-tkg-captioning/models/qwen2.5-14b'
config={'schema_version':1,'path_policy':'Paths are expanded on the server; ${REMOTE_HOME} means pathlib.Path.home(). No global environment settings changed.','qwen14b':{'model_id':'Qwen/Qwen2.5-14B-Instruct','revision':'cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8','path':alias(qwen)},'clip':{'model_id':'openai/clip-vit-base-patch32','path':alias(shared/'models/clip-vit-base-patch32')},'datasets':datasets,'caption_reference_policy':'References are evaluation-only. Never inject them into graph-to-text prompts. Flickr full test remains blocked until all required images exist.'}
(shared/'ASSET_REGISTRY.json').write_text(json.dumps(config,indent=2))
(home/'rsg-former/reports/SHARED_ASSET_REGISTRY.json').write_text(json.dumps(config,indent=2))
result={'at_utc':datetime.now(timezone.utc).isoformat(),'status':'EXISTING_ASSETS_SHARED','events':events,'datasets':datasets,'asset_registry':config,'free_bytes':os.statvfs(home).f_bavail*os.statvfs(home).f_frsize,'downloaded':False,'VG_assets_untouched':True}
(shared/'maintenance/KARPATHY_MOVE_RESULT_20261010.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
