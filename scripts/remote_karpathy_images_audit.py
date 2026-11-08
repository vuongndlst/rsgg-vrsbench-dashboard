"""Read-only discovery and exact Karpathy-test image coverage; no captions generated."""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter,defaultdict
import os,json,re,subprocess,hashlib
home=Path.home()
def alias(x):return re.sub(r'/(home|users)/[^/\s]+',r'/\1/${ACCOUNT}',str(x).replace(str(home),'${REMOTE_HOME}').replace(home.name,'${ACCOUNT}'))
coco=json.loads((home/'skgcap_vlm/data/dataset_coco.json').read_text());flickr=json.loads((home/'skgcap_vlm/data/dataset_flickr30k.json').read_text())
tests={'coco':[i for i in coco['images'] if i['split']=='test'],'flickr30k':[i for i in flickr['images'] if i['split']=='test']}
names={k:{i['filename'] for i in v} for k,v in tests.items()};lookup={name:k for k,items in names.items() for name in items}
args=['find','/','(','-path','/proc','-o','-path','/sys','-o','-path','/dev','-o','-path','/run','-o','-name','.git','-o','-name','node_modules','-o','-name','.venv','-o','-name','site-packages','-o','-name','dist-packages',')','-prune','-o','(','-type','f','-o','-type','l',')','(','-iname','*.jpg','-o','-iname','*.jpeg',')','-print0']
p=subprocess.Popen(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
try:out,err=p.communicate(timeout=150);completed=True
except subprocess.TimeoutExpired:p.kill();out,err=p.communicate();completed=False
found=defaultdict(list);counts=Counter();total=0
for entry in out.split(b'\0'):
    if not entry:continue
    path=Path(os.fsdecode(entry));total+=1
    if path.name in lookup:
        try:
            if path.is_file() and path.stat().st_size>0:
                found[lookup[path.name]].append({'filename':path.name,'path':alias(path),'parent':alias(path.parent),'bytes':path.stat().st_size})
        except OSError:pass
    if any(s in str(path).lower() for s in ['coco','flickr','karpathy']):counts[alias(path.parent)]+=1
datasets={}
for k,records in tests.items():
    present={x['filename'] for x in found[k]};datasets[k]={'expected_test_images':len(records),'present_unique_test_images':len(present),'missing_count':len(names[k]-present),'missing_filenames':sorted(names[k]-present),'matching_files':found[k],'matched_parent_counts':dict(Counter(x['parent'] for x in found[k]))}
print(json.dumps({'at_utc':datetime.now(timezone.utc).isoformat(),'scan_completed':completed,'find_returncode':p.returncode,'permission_error_count':len(err.splitlines()),'directory_symlinks_followed':False,'jpeg_files_examined':total,'image_parent_counts':dict(counts),'datasets':datasets,'note':'Filename existence and nonzero-size coverage only; no full JPEG decode/CRC; no evaluation or GPU use.'}))
