"""Execute only explicitly authorized model removals, preserving one shared CLIP and Qwen.

Linux renameat2 exchange atomically replaces the old source by a compatibility
symlink while moving its directory. Existing readers/writers never lose the old
path. No process is signalled; no training configuration or dataset is changed.
"""
from pathlib import Path
from datetime import datetime,timezone
import os,json,re,hashlib,shutil,ctypes,struct
home=Path.home().resolve(); shared=home/'shared-assets';events=[]
def alias(x):return re.sub(r'/(home|users)/[^/\s]+',r'/\1/${ACCOUNT}',str(x).replace(str(home),'${REMOTE_HOME}').replace(home.name,'${ACCOUNT}'))
def boundary(p):
    assert p.is_absolute() and p!=home and p.is_relative_to(home)
    assert p.parent.resolve().is_relative_to(home),'parent_outside_home'
    if not p.is_symlink():assert p.resolve().is_relative_to(home),'target_outside_home'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()
def journal(e):
    events.append(e)
    (shared/'maintenance/MODEL_CLEANUP_20261010.json').write_text(json.dumps({'at_utc':datetime.now(timezone.utc).isoformat(),'events':events},indent=2))
def consumers(root):
    refs=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:
            if proc.stat().st_uid!=os.getuid() or int(proc.name)==os.getpid():continue
            cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
            if str(root) in cmd:refs.append(int(proc.name));continue
            paths=[]
            try:
                for fd in (proc/'fd').iterdir():
                    try:paths.append(os.readlink(fd))
                    except OSError:pass
            except OSError:pass
            try:
                for line in (proc/'maps').read_text().splitlines():
                    fields=line.split(maxsplit=5)
                    if len(fields)==6:paths.append(fields[-1])
            except OSError:pass
            if any(x==str(root) or x.startswith(str(root)+'/') for x in paths):refs.append(int(proc.name))
        except OSError:pass
    return sorted(set(refs))
def move_with_alias(source,dest):
    boundary(source);boundary(dest)
    if source.is_symlink() and source.resolve()==dest and dest.is_dir():
        journal({'action':'already_shared','source':alias(source),'destination':alias(dest)});return
    assert source.is_dir() and not source.is_symlink() and not os.path.lexists(dest)
    assert source.stat().st_uid==os.getuid()
    dest.parent.mkdir(parents=True,exist_ok=True)
    assert source.stat().st_dev==dest.parent.stat().st_dev,'cross_filesystem_move_forbidden'
    before=(source.stat().st_dev,source.stat().st_ino)
    libc=ctypes.CDLL(None,use_errno=True)
    exchange=libc.renameat2;exchange.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint];exchange.restype=ctypes.c_int
    dest.symlink_to(dest,target_is_directory=True)
    journal({'action':'move_prepared','source':alias(source),'destination':alias(dest),'inode_before':list(before)})
    rc=exchange(-100,os.fsencode(source),-100,os.fsencode(dest),2)
    if rc:
        err=ctypes.get_errno();dest.unlink();raise OSError(err,'atomic_exchange_failed')
    assert source.is_symlink() and source.resolve()==dest
    assert before==(dest.stat().st_dev,dest.stat().st_ino)
    journal({'action':'moved_and_alias_verified','source':alias(source),'destination':alias(dest),'same_inode':True})
def remove(relative,compat=None):
    p=home/relative;boundary(p)
    if not os.path.lexists(p):journal({'action':'already_absent','path':alias(p)});return
    if compat and p.is_symlink() and p.resolve()==compat.resolve():
        journal({'action':'already_compatibility_alias','path':alias(p)});return
    assert p.lstat().st_uid==os.getuid(),'not_owned'
    active=consumers(p)
    if active:
        journal({'action':'skipped_active_consumer','path':alias(p),'pids':active});return
    journal({'action':'delete_prepared','path':alias(p),'resolved_path':alias(p.resolve())})
    if p.is_symlink() or p.is_file():p.unlink()
    else:shutil.rmtree(p)
    assert not os.path.lexists(p)
    if compat:p.symlink_to(compat,target_is_directory=True)
    journal({'action':'removed','path':alias(p),'compatibility_alias':alias(compat) if compat else None})

boundary(shared);shared.mkdir(exist_ok=True);(shared/'maintenance').mkdir(exist_ok=True)
free_before=os.statvfs(home).f_bavail*os.statvfs(home).f_frsize
clip_src=home/'coco_kb_v3/runtime/models/clip-vit-base-patch32';clip=shared/'models/clip-vit-base-patch32'
base=clip if clip.is_dir() else clip_src
required=['config.json','preprocessor_config.json','pytorch_model.bin','tokenizer_config.json','tokenizer.json','vocab.json','merges.txt']
assert all((base/n).is_file() for n in required),'incomplete_clip'
assert json.loads((base/'config.json').read_text())['model_type']=='clip'
clip_sha=sha(base/'pytorch_model.bin')
other=home/'etechs/final_project/var/models/clip-vit-base-patch32'
if other.is_dir() and not other.is_symlink():assert sha(other/'pytorch_model.bin')==clip_sha,'clip_copy_differs'
move_with_alias(clip_src,clip)
assert sha(clip/'pytorch_model.bin')==clip_sha

qrel='fair-project/models/hf_cache/hub/models--Qwen--Qwen2.5-14B-Instruct';qsrc=home/qrel;qdest=shared/'models/huggingface/hub/models--Qwen--Qwen2.5-14B-Instruct'
assert not consumers(qsrc),'qwen_consumer_still_active_defer_move'
revision='cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8';snap=qsrc/'snapshots'/revision
idx=json.loads((snap/'model.safetensors.index.json').read_text());names=sorted(set(idx['weight_map'].values()));assert len(names)==8
before={n:(snap/n).stat() for n in names};index_sha=sha(snap/'model.safetensors.index.json')
move_with_alias(qsrc,qdest)
for n,s in before.items():
    p=qdest/'snapshots'/revision/n;t=p.stat()
    assert (t.st_dev,t.st_ino,t.st_size)==(s.st_dev,s.st_ino,s.st_size),'qwen_shard_changed'
    with p.open('rb') as f:
        size=struct.unpack('<Q',f.read(8))[0];assert 0<size<16*1024**2;h=json.loads(f.read(size))
    assert t.st_size==8+size+max(v['data_offsets'][1] for k,v in h.items() if k!='__metadata__')
assert sha(qdest/'snapshots'/revision/'model.safetensors.index.json')==index_sha

# Shared lightweight HF repository points to the single CLIP directory.
clip_repo=shared/'models/huggingface/hub/models--openai--clip-vit-base-patch32';clip_revision='3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268'
(clip_repo/'snapshots').mkdir(parents=True,exist_ok=True);(clip_repo/'refs').mkdir(exist_ok=True)
link=clip_repo/'snapshots'/clip_revision
if not os.path.lexists(link):link.symlink_to(clip,target_is_directory=True)
assert link.resolve()==clip
(clip_repo/'refs/main').write_text(clip_revision)
targets=[
 ('.cache/clip/ViT-B-32.pt',None),
 ('.cache/huggingface/hub/hub/models--lmms-lab--llava-onevision-qwen2-7b-ov',None),
 ('.cache/huggingface/hub/models--google--siglip-so400m-patch14-384',None),
 ('.cache/huggingface/hub/models--openai--clip-vit-base-patch32',clip_repo),
 ('etechs/final_project/var/models/clip-vit-base-patch32',clip),
 ('etechs/ouc_stkg_captioning_research_pack/artifacts/multimodal-labels-v1/models/Qwen2.5-VL-3B-Instruct',None),
 ('fair-project/models/hf_cache/hub/models--InstantX--SD3-Controlnet-Canny',None),
 ('fair-project/models/hf_cache/hub/models--lmms-lab--llava-onevision-qwen2-7b-ov',None),
 ('fair-project/models/hf_cache/hub/models--openai--clip-vit-base-patch32',clip_repo),
 ('fair-project/models/hf_cache/hub/models--stabilityai--stable-diffusion-3-medium-diffusers',None),
 ('fair-project/models/hf_cache/models--google--siglip-so400m-patch14-384',None),
 ('v2_kb_20260909/cache/huggingface/models--openai--clip-vit-large-patch14-336',None),
]
for relative,compat in targets:remove(relative,compat)
free_after=os.statvfs(home).f_bavail*os.statvfs(home).f_frsize
result={'at_utc':datetime.now(timezone.utc).isoformat(),'status':'COMPLETE' if not any(e['action']=='skipped_active_consumer' for e in events) else 'PARTIAL_ACTIVE_CONSUMER','events':events,'clip_weight_sha256':clip_sha,'clip_shared_path':alias(clip),'qwen_shared_snapshot':alias(qdest/'snapshots'/revision),'qwen_shard_inodes_preserved':8,'qwen_index_sha256':index_sha,'free_before_bytes':free_before,'free_after_bytes':free_after,'observed_free_gain_bytes':free_after-free_before,'processes_signalled':False,'VG_code_config_data_untouched':True}
(shared/'maintenance/MODEL_CLEANUP_RESULT_20261010.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
