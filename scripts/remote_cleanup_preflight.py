"""Read-only preflight for the user's exact model cleanup and shared asset moves."""
from pathlib import Path
from datetime import datetime,timezone
import os,json,subprocess,hashlib,re
home=Path.home();shared=home/'shared-assets'
targets=[
 '.cache/clip/ViT-B-32.pt',
 '.cache/huggingface/hub/hub/models--lmms-lab--llava-onevision-qwen2-7b-ov',
 '.cache/huggingface/hub/models--google--siglip-so400m-patch14-384',
 '.cache/huggingface/hub/models--openai--clip-vit-base-patch32',
 'coco_kb_v3/runtime/models/clip-vit-base-patch32',
 'etechs/ouc_stkg_captioning_research_pack/artifacts/multimodal-labels-v1/models/Qwen2.5-VL-3B-Instruct',
 'fair-project/models/hf_cache/hub/models--InstantX--SD3-Controlnet-Canny',
 'fair-project/models/hf_cache/hub/models--lmms-lab--llava-onevision-qwen2-7b-ov',
 'fair-project/models/hf_cache/hub/models--openai--clip-vit-base-patch32',
 'fair-project/models/hf_cache/hub/models--stabilityai--stable-diffusion-3-medium-diffusers',
 'fair-project/models/hf_cache/models--google--siglip-so400m-patch14-384',
 'v2_kb_20260909/cache/huggingface/models--openai--clip-vit-large-patch14-336',
]
moves=[('etechs/final_project/var/models/clip-vit-base-patch32','models/clip-vit-base-patch32'),('fair-project/models/hf_cache/hub/models--Qwen--Qwen2.5-14B-Instruct','models/huggingface/hub/models--Qwen--Qwen2.5-14B-Instruct')]
def alias(p):return re.sub(r'/(home|users)/[^/\s]+',r'/\1/${ACCOUNT}',str(p).replace(str(home),'${REMOTE_HOME}').replace(home.name,'${ACCOUNT}'))
def info(p):
    present=os.path.lexists(p);row={'path':alias(p),'present':present}
    if present:
        s=p.lstat();row.update(symlink=p.is_symlink(),real_path=alias(p.resolve()),inode_key=f'{s.st_dev}:{s.st_ino}',owner_is_current=s.st_uid==os.getuid())
        r=subprocess.run(['du','-x','-s','-B1',str(p)],capture_output=True,text=True,timeout=40)
        row['allocated_bytes']=int(r.stdout.split()[0]) if r.stdout else None
        row['du_returncode']=r.returncode
        if p.is_dir():row['top_entries']=sorted(x.name for x in p.iterdir())[:80]
    return row
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()
records=[info(home/p) for p in targets];moving=[]
for source,dest in moves:
    row=info(home/source);row['destination']=info(shared/dest);moving.append(row)
clip=home/moves[0][0]
clip_files={p.name:{'bytes':p.stat().st_size,'sha256':digest(p)} for p in clip.iterdir() if p.is_file() and (p.stat().st_size<20*1024**2 or p.name=='pytorch_model.bin')}
qwen=home/moves[1][0]/'snapshots/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8'
index=json.loads((qwen/'model.safetensors.index.json').read_text())
shards=[{'name':n,'bytes':(qwen/n).stat().st_size,'inode_key':f'{(qwen/n).stat().st_dev}:{(qwen/n).stat().st_ino}','real_path':alias((qwen/n).resolve())} for n in sorted(set(index['weight_map'].values()))]
blockers=[];procs=[]
watched=[home/p for p in targets]+[home/s for s,d in moves]
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit():continue
    try:
        if proc.stat().st_uid!=os.getuid() or int(proc.name)==os.getpid():continue
        cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
        paths=[]
        try:
            for f in (proc/'fd').iterdir():
                try:paths.append(os.readlink(f))
                except OSError:pass
        except OSError:pass
        try:
            for line in (proc/'maps').read_text().splitlines():
                fields=line.split(maxsplit=5)
                if len(fields)==6:paths.append(fields[-1])
        except OSError:pass
        for root in watched:
            hits=[alias(p) for p in paths if p==str(root) or p.startswith(str(root)+'/')]
            if str(root) in cmd or hits:blockers.append({'pid':int(proc.name),'target':alias(root),'open_or_mapped_paths':hits[:10],'explicit_argument_reference':str(root) in cmd})
    except OSError:pass
print(json.dumps({'at_utc':datetime.now(timezone.utc).isoformat(),'delete_targets':records,'moves':moving,'clip_files':clip_files,'qwen_shards':shards,'active_target_consumers':blockers,'free_bytes':os.statvfs(home).f_bavail*os.statvfs(home).f_frsize,'shared_root':info(shared),'gpu_work_started':False}))
