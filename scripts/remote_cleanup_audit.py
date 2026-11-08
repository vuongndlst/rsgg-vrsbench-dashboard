"""Read-only size, consumer, dataset and backup audit for authorized cleanup."""
from pathlib import Path
from datetime import datetime, timezone
import os,json,re,subprocess,time,pwd
home=Path.home(); start=time.monotonic()
def alias(x):
    return re.sub(r'/(home|users)/[^/\s]+',r'/\1/${ACCOUNT}',str(x).replace(str(home),'${REMOTE_HOME}').replace(home.name,'${ACCOUNT}').replace(pwd.getpwuid(os.getuid()).pw_name,'${ACCOUNT}'))
def run(args,timeout=180):
    try:
        p=subprocess.run(args,capture_output=True,timeout=timeout)
        return p.returncode,p.stdout,p.stderr
    except subprocess.TimeoutExpired as e:return 'timeout',e.stdout or b'',b''
def sizes(p,depth):
    code,out,err=run(['du','-x','-B1',f'--max-depth={depth}',str(p)])
    rows=[]
    for line in os.fsdecode(out).splitlines():
        try:n,path=line.split('\t',1);rows.append({'path':alias(path),'allocated_bytes':int(n)})
        except ValueError:pass
    return {'root':alias(p),'returncode':code,'rows':sorted(rows,key=lambda r:r['allocated_bytes'],reverse=True),'permission_errors':len(err.splitlines())}
own=[]; other_count=0
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit():continue
    try:
        if proc.stat().st_uid!=os.getuid():other_count+=1;continue
        args=(proc/'cmdline').read_bytes().split(b'\0'); args=[os.fsdecode(a) for a in args if a]
        if not args:continue
        raw=' '.join(args)
        matches=[k for k in ['clip','siglip','llava','diffusion','controlnet','Qwen2.5-VL-3B','fair-project','rsg-former','karpathy','coco','flickr','backup','download'] if k.lower() in raw.lower()]
        paths=set(); errors=0
        for folder in [proc/'fd']:
            try:
                for f in folder.iterdir():
                    try:
                        t=os.readlink(f)
                        if t.startswith(str(home)):paths.add(alias(t))
                    except OSError:errors+=1
            except OSError:errors+=1
        try:cwd=alias(os.readlink(proc/'cwd'))
        except OSError:cwd='unreadable'
        try:
            for line in (proc/'maps').read_text().splitlines():
                fields=line.split(maxsplit=5)
                if len(fields)==6 and fields[-1].startswith(str(home)):paths.add(alias(fields[-1]))
        except OSError:errors+=1
        own.append({'pid':int(proc.name),'executable':alias(Path(args[0]).name),'cwd':cwd,'argument_keyword_matches':matches,'open_or_mapped_home_paths':sorted(paths)[:150],'read_errors':errors})
    except OSError:pass
pruned=['.git','.venv','node_modules','__pycache__','site-packages','dist-packages','.ssh','.aws','blobs']
args=['find',str(home),'(']
for i,n in enumerate(pruned):
    if i:args+=['-o']
    args+=['-name',n]
args+=[')','-prune','-o','(']
patterns=['*karpathy*','dataset_coco.json','dataset_flickr*.json','*captions*.json','*captions*.zip','*flickr*','*Flickr*','*coco*.zip','*COCO*.zip','*backup*','*Backup*','*.bak','*.old','*archive*','*Archive*']
for i,n in enumerate(patterns):
    if i:args+=['-o']
    args+=['-name',n]
args+=[')','-print0']
code,out,err=run(args,150);discovery=[]
for entry in out.split(b'\0'):
    if not entry:continue
    p=Path(os.fsdecode(entry))
    try:
        s=p.lstat();discovery.append({'path':alias(p),'directory':p.is_dir(),'symlink':p.is_symlink(),'bytes':s.st_size,'mtime_utc':datetime.fromtimestamp(s.st_mtime,timezone.utc).isoformat()})
    except OSError:pass
summary={'at_utc':datetime.now(timezone.utc).isoformat(),'mode':'read_only','own_processes':own,'other_account_process_count':other_count,'consumer_limit':'Own processes: argument keywords, cwd, fd and mmap; absence does not prove no consumer. Other accounts not inspected.','root_entries':[{'path':alias(p),'directory':p.is_dir(),'symlink':p.is_symlink()} for p in Path('/').iterdir()],'discovery':{'returncode':code,'matches':discovery,'permission_errors':len(err.splitlines())},'home_sizes':sizes(home,1),'fair_sizes':sizes(home/'fair-project',2)}
for label,base in [('datasets',home/'fair-project/datasets'),('data',home/'fair-project/data'),('models',home/'fair-project/models'),('tmp',Path('/tmp'))]:
    if base.is_dir():summary[label+'_sizes']=sizes(base,2 if label!='tmp' else 1)
summary['wall_seconds']=time.monotonic()-start
summary['free_bytes']=os.statvfs(home).f_bavail*os.statvfs(home).f_frsize
print(json.dumps(summary,ensure_ascii=False))
