"""Authorized P0 preparation on SSH: project-owned dependencies and missing metadata."""
import os,sys,json,subprocess,urllib.request,hashlib,zipfile,venv,time
from pathlib import Path
HOME=Path.home();ROOT=HOME/'rsg-former';ROOT.mkdir(exist_ok=True)
for name in ['data/vg150','data/star','artifacts','reports','weights','.tools']:(ROOT/name).mkdir(parents=True,exist_ok=True)
def say(event,**kw):print(json.dumps({'event':event,**kw}),flush=True)
free=os.statvfs(ROOT).f_bavail*os.statvfs(ROOT).f_frsize
assert free>8*1024**3,'insufficient_storage'
say('storage_verified',free_gib=round(free/1024**3,2),project_root='${REMOTE_HOME}/rsg-former')
ENV=ROOT/'.venv'
venv.EnvBuilder(system_site_packages=True,with_pip=False).create(ENV)
PY=ENV/'bin/python';DEPS=ROOT/'.tools/deps';DEPS.mkdir(exist_ok=True)
env=dict(os.environ,PIP_DISABLE_PIP_VERSION_CHECK='1',PYTHONDONTWRITEBYTECODE='1',PIP_CACHE_DIR=str(ROOT/'.tools/pip-cache'))
# Additive overlay preserves the existing PyTorch/CUDA stack and never updates user-site packages.
check=subprocess.run([str(PY),'-c','import h5py;print(h5py.__version__)'],env=dict(env,PYTHONPATH=str(DEPS)),capture_output=True,text=True)
if check.returncode:
    install=subprocess.run(['/usr/bin/python3','-m','pip','install','--target',str(DEPS),'--no-deps','h5py==3.15.1'],env=env,capture_output=True,text=True)
    if install.returncode:
        say('dependency_install_failed',exit_code=install.returncode);sys.exit(1)
say('project_environment_ready',python_alias='${REMOTE_HOME}/rsg-former/.venv/bin/python',isolation='project_overlay_inherits_locked_system_torch; no shared packages modified')
def small_json(url):
    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'RSG-Former-P0'}),timeout=30) as r:return json.loads(r.read(4*1024*1024))
def download(url,target,expected=None,max_bytes=300*1024*1024):
    if target.exists():
        digest=hashlib.sha256(target.read_bytes()).hexdigest()
        if expected is None or digest==expected:return {'file':target.name,'bytes':target.stat().st_size,'sha256':digest,'status':'reused','source':url}
        raise RuntimeError('existing_file_hash_mismatch')
    request=urllib.request.Request(url,headers={'User-Agent':'RSG-Former-P0'})
    with urllib.request.urlopen(request,timeout=60) as response:
        length=int(response.headers.get('Content-Length',0));assert length<=max_bytes
        total=0;digest=hashlib.sha256();part=target.with_name(target.name+'.partial')
        with part.open('wb') as f:
            while True:
                block=response.read(1024*1024)
                if not block:break
                total+=len(block);assert total<=max_bytes
                f.write(block);digest.update(block)
    value=digest.hexdigest()
    if expected and value!=expected:raise RuntimeError('download_hash_mismatch')
    part.replace(target)
    say('download_complete',file=target.name,bytes=total)
    return {'file':target.name,'bytes':total,'sha256':value,'status':'downloaded','source':url}
manifest={'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'files':[],'environment':'project_overlay_with_locked_shared_torch','images_extracted':False}
sha='ceb71fa88461c2a97a6258a80f47669d89207296';base='https://raw.githubusercontent.com/KaihuaTang/Scene-Graph-Benchmark.pytorch/'+sha+'/'
for name in ['VG-SGG-dicts-with-attri.json','image_data.json']:
    manifest['files'].append(download(base+'datasets/vg/'+name,ROOT/'data/vg150'/name,max_bytes=32*1024*1024))
# HDF5 published by the LLM4SGG authors; pin their release and advertised LFS digest.
hf=small_json('https://huggingface.co/api/datasets/kb-kim/LLM4SGG?blobs=true');item=next(x for x in hf['siblings'] if x['rfilename']=='VG-SGG-with-attri.h5')
url='https://huggingface.co/datasets/kb-kim/LLM4SGG/resolve/'+hf['sha']+'/VG-SGG-with-attri.h5'
manifest['files'].append(download(url,ROOT/'data/vg150/VG-SGG-with-attri.h5',item['lfs']['sha256']))
manifest['vg_hdf5_source']={'publisher':'LLM4SGG authors; mirror of VG150 metadata','revision':hf['sha'],'original_toolkit_equivalence':'requires schema/split/content validation, not inferred from filename'}
star=small_json('https://huggingface.co/api/datasets/Zhuzi24/STAR?blobs=true');annotation=next(x for x in star['siblings'] if x['rfilename']=='STAR_SGG_Annotation.zip')
manifest['star']={'revision':star['sha'],'card':star.get('cardData',{}),'files':[{'name':x['rfilename'],'bytes':x.get('size',x.get('lfs',{}).get('size')),'sha256':x.get('lfs',{}).get('sha256')} for x in star['siblings']],'image_download':'deferred_capacity_insufficient; P0 feasibility audit only'}
manifest['files'].append(download('https://huggingface.co/datasets/Zhuzi24/STAR/resolve/'+star['sha']+'/STAR_SGG_Annotation.zip',ROOT/'data/star/STAR_SGG_Annotation.zip',annotation['lfs']['sha256'],max_bytes=100*1024*1024))
with zipfile.ZipFile(ROOT/'data/star/STAR_SGG_Annotation.zip') as z:
    manifest['star']['annotation_members']=[{'name':i.filename,'bytes':i.file_size,'crc32':f'{i.CRC:08x}'} for i in z.infolist()][:120]
    manifest['star']['annotation_uncompressed_bytes']=sum(i.file_size for i in z.infolist())
(ROOT/'reports/P0_PREPARATION.json').write_text(json.dumps(manifest,indent=2))
say('p0_preparation_complete',files=len(manifest['files']),report='reports/P0_PREPARATION.json')
