"""Read-only focused dataset, backup, project and old-process audit."""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import os,json,subprocess,re,hashlib
home=Path.home()
def alias(x):return re.sub(r'/(home|users)/[^/\s]+',r'/\1/${ACCOUNT}',str(x).replace(str(home),'${REMOTE_HOME}').replace(home.name,'${ACCOUNT}'))
def du(p,depth):
    r=subprocess.run(['du','-x','-B1',f'--max-depth={depth}',str(p)],capture_output=True,text=True,timeout=160)
    return {'returncode':r.returncode,'rows':[{'path':alias(s.split('\t',1)[1]),'bytes':int(s.split('\t',1)[0])} for s in r.stdout.splitlines() if '\t' in s]}
roots=['etechs','skgcap_vlm/data','skgcap_vlm/kgproj/datasets','coco_kb_v3/data','image-captioning-kg/datasets','fair-project/datasets','disk-cleanup-20260925','v2_kb','v2_kb_20260909','spice_setup']
dirs=[]
for r in roots:
    p=home/r
    if p.is_dir():
        entries=[]
        for x in p.iterdir():
            if x.name.startswith('.') or x.name in ['node_modules','__pycache__']:continue
            entries.append({'name':x.name,'directory':x.is_dir(),'symlink':x.is_symlink(),'target':alias(x.resolve()) if x.is_symlink() else None,'bytes':x.lstat().st_size})
        dirs.append({'path':alias(p),'entries':entries[:250],'sizes':du(p,2 if r=='etechs' else 1)})
metadata=[]
for rel in ['skgcap_vlm/data/dataset_coco.json','skgcap_vlm/data/dataset_flickr30k.json','coco_kb_v3/data/dataset_coco.json']:
    p=home/rel
    if not p.is_file():continue
    raw=p.read_bytes();d=json.loads(raw);images=d.get('images',[])
    metadata.append({'path':alias(p),'sha256':hashlib.sha256(raw).hexdigest(),'keys':list(d),'image_count':len(images),'splits':dict(Counter(i.get('split') for i in images)),'image_fields':list(images[0]) if images else [],'test_samples':[{k:i.get(k) for k in ['filename','filepath','imgid','cocoid','split']} for i in images if i.get('split')=='test'][:3]})
processes=[]
uptime=float(Path('/proc/uptime').read_text().split()[0]); hz=os.sysconf('SC_CLK_TCK')
for pid in [2916729,2906059,648773]:
    p=Path('/proc')/str(pid)
    if not p.exists():continue
    cmd=[os.fsdecode(x) for x in (p/'cmdline').read_bytes().split(b'\0') if x]
    # Only executable/script names and known switch names, never option values.
    scripts=[Path(a).name for a in cmd if a.endswith(('.py','.sh'))]
    st=(p/'stat').read_text().rsplit(')',1)[1].split();deleted={}
    for f in (p/'fd').iterdir():
        try:
            link=os.readlink(f);s=f.stat()
            if link.endswith(' (deleted)'):
                key=f'{s.st_dev}:{s.st_ino}';deleted[key]={'path':alias(link),'bytes':s.st_size,'allocated_bytes':s.st_blocks*512}
        except OSError:pass
    processes.append({'pid':pid,'scripts':scripts,'state':st[0],'age_seconds':uptime-int(st[19])/hz,'rss_bytes':int(st[21])*os.sysconf('SC_PAGE_SIZE'),'deleted_open_files':list(deleted.values()),'deleted_open_allocated_bytes':sum(i['allocated_bytes'] for i in deleted.values()),'io_counters':dict(line.split(': ',1) for line in (p/'io').read_text().splitlines())})
backup_args=['find',str(home),'-maxdepth','6','(','-name','.git','-o','-name','node_modules','-o','-name','.venv','-o','-name','site-packages','-o','-name','dist-packages',')','-prune','-o','-type','d','(','-iname','*backup*','-o','-iname','*archive*','-o','-iname','*old*',')','-print0']
r=subprocess.run(backup_args,capture_output=True,timeout=60);backups=[]
for x in r.stdout.split(b'\0'):
    if x:
        p=Path(os.fsdecode(x))
        try:backups.append({'path':alias(p),'sizes':du(p,0)})
        except (OSError,subprocess.TimeoutExpired):pass
print(json.dumps({'at_utc':datetime.now(timezone.utc).isoformat(),'directories':dirs,'karpathy_metadata':metadata,'old_processes':processes,'backup_candidates':backups},ensure_ascii=False))
