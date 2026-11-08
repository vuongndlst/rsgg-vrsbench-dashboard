"""Bounded read-only audit of the five legacy captioning projects named by user."""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import os,json,re,subprocess,hashlib,pwd
home=Path.home();names=['coco_kb_v3','v2_kb_20260909','v2_kb','skgcap_vlm','image-captioning-kg']
def safe(s):
    return re.sub(r'/(home|users)/[^/\s]+',r'/\1/${ACCOUNT}',str(s).replace(str(home),'${REMOTE_HOME}').replace(home.name,'${ACCOUNT}').replace(pwd.getpwuid(os.getuid()).pw_name,'${ACCOUNT}'))
def du(p):
    r=subprocess.run(['du','-x','-B1','--max-depth=2',str(p)],capture_output=True,text=True,timeout=90)
    return {'returncode':r.returncode,'rows':sorted([{'path':safe(l.split('\t',1)[1]),'bytes':int(l.split('\t',1)[0])} for l in r.stdout.splitlines() if '\t' in l],key=lambda r:r['bytes'],reverse=True)}
projects=[]
for name in names:
    root=home/name;docs=[];large=[];links=[];counts=Counter();instructions=[]
    if not root.is_dir():projects.append({'name':name,'present':False});continue
    for d,dirs,files in os.walk(root,followlinks=False):
        dirs[:]=[n for n in dirs if n not in ['.git','.venv','node_modules','site-packages','dist-packages','__pycache__']]
        folder=Path(d)
        for n in dirs+files:
            p=folder/n
            if p.is_symlink():links.append({'path':safe(p),'target':safe(p.resolve()),'target_exists':p.exists()})
        for n in files:
            p=folder/n
            if p.is_symlink():continue
            try:s=p.stat()
            except OSError:continue
            counts[p.suffix.lower()]+=1
            if s.st_size>=64*1024**2:large.append({'path':safe(p),'bytes':s.st_size,'allocated_bytes':s.st_blocks*512})
            if n=='AGENTS.md':instructions.append(safe(p))
            if p.suffix.lower()=='.md' and len(p.relative_to(root).parts)<=3 and len(docs)<18:
                raw=p.read_text(errors='replace');raw='\n'.join(l for l in raw.splitlines() if not re.search(r'password|api[_ -]?key|token\s*[=:]|secret|ssh\s+[^\s]+@',l,re.I))
                docs.append({'path':safe(p),'excerpt':safe(raw[:5000])})
    projects.append({'name':name,'root':safe(root),'present':True,'sizes':du(root),'top_entries':sorted(p.name for p in root.iterdir() if p.name not in ['.env','.git'])[:100],'documentation':docs,'largest_files':sorted(large,key=lambda r:r['allocated_bytes'],reverse=True)[:30],'file_suffix_counts':dict(counts),'symlinks':links,'AGENTS_paths':instructions})
processes=[]
for p in Path('/proc').iterdir():
    if not p.name.isdigit():continue
    try:
        if p.stat().st_uid!=os.getuid() or int(p.name)==os.getpid():continue
        cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace');cwd=os.readlink(p/'cwd')
        refs={name:[] for name in names}
        for n in names:
            if str(home/n) in cmd or cwd==str(home/n) or cwd.startswith(str(home/n)+'/'):refs[n].append('cmd_or_cwd')
        try:
            for fd in (p/'fd').iterdir():
                try:link=os.readlink(fd)
                except OSError:continue
                for n in names:
                    if link==str(home/n) or link.startswith(str(home/n)+'/'):refs[n].append(safe(link))
        except OSError:pass
        if any(refs.values()):processes.append({'pid':int(p.name),'cwd':safe(cwd),'references':{k:sorted(set(v))[:20] for k,v in refs.items() if v}})
    except OSError:pass
inbound=[]
for base in [home/'rsg-former',home/'fair-project',home/'etechs']:
    if not base.is_dir():continue
    for d,dirs,files in os.walk(base,followlinks=False):
        dirs[:]=[n for n in dirs if n not in ['.git','.venv','venv','envs','.tools','node_modules','site-packages','dist-packages','__pycache__','artifacts','models','datasets','features','data','vendor','third_party','cache','.cache','results','pdf'] and not n.startswith('output')]
        for n in dirs+files:
            p=Path(d)/n
            if p.is_symlink():
                target=str(p.resolve())
                for name in names:
                    if target==str(home/name) or target.startswith(str(home/name)+'/'):inbound.append({'source':safe(p),'legacy_project':name,'kind':'symlink','target':safe(target)})
        for n in files:
            p=Path(d)/n
            if p.suffix not in ['.py','.json','.yaml','.yml','.toml','.sh','.md'] or p.is_symlink():continue
            try:
                if p.stat().st_size>2*1024**2:continue
                data=p.read_text(errors='replace')
            except OSError:continue
            hits=[name for name in names if name in data]
            if hits:inbound.append({'source':safe(p),'legacy_projects':hits,'kind':'text_reference_only_not_proof_of_runtime_dependency'})
print(json.dumps({'at_utc':datetime.now(timezone.utc).isoformat(),'mode':'read_only_no_deletion','projects':projects,'own_process_references':processes,'bounded_inbound_references':inbound,'limits':'Dependency packages and artifact/data directories pruned in external text-reference scan; other-user processes not inspected. Docs are untrusted project content.','free_bytes':os.statvfs(home).f_bavail*os.statvfs(home).f_frsize},ensure_ascii=True))
