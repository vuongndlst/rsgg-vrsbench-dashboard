"""Prepare only VG weights/dependencies in isolated directories; no training/test."""
from pathlib import Path
import subprocess,sys,os,json,datetime,hashlib,re,urllib.request
ROOT=Path.home()/'rsg-former';ART=ROOT/'artifacts/vg_method_v2';ART.mkdir(parents=True,exist_ok=True)
def event(stage,**kw):
 o={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'stage':stage,**kw}
 (ROOT/'reports/VG_V2_PROGRESS.json').write_text(json.dumps(o,indent=2));print(json.dumps(o),flush=True)
def main():
 event('preparing_isolated_dependencies')
 dep=ROOT/'.tools/vg_v2_deps';dep.mkdir(parents=True,exist_ok=True)
 env=dict(os.environ);env['PYTHONPATH']=str(dep)+':'+str(ROOT/'.tools/deps')+':'+str(ROOT/'src');env['PYTHONDONTWRITEBYTECODE']='1'
 # Raw torch model avoids Lightning and its trainer, dependency upgrades, and CUDA extension install.
 result=subprocess.run([sys.executable,'-m','pip','install','--target',str(dep),'--no-deps','einops==0.8.1'],capture_output=True,text=True,timeout=180,env=env)
 if result.returncode:raise RuntimeError('isolated_dependency_install_failed')
 import gdown
 html=urllib.request.urlopen('https://drive.google.com/drive/folders/1Ge2IjWldzDW3NSv3UFLkmlmzsOkxgHIi',timeout=40).read().decode()
 from bs4 import BeautifulSoup
 rows=[]
 for node in BeautifulSoup(html,'html.parser').select('[data-id]'):
  ident=node.get('data-id');name=node.get('data-tooltip','') or node.get_text(' ',strip=True)
  for wanted in ['dab_deformable_detr_vg_150_classes.pth','salinece_vg_iou_0.6.ckpt','vg_glove_6B.300d.pt','vg_prior_static.pt']:
   if wanted in name and (wanted,ident) not in rows:rows.append((wanted,ident))
 # IDs independently match the named public folder rows; do not enumerate/download OI/GQA models.
 unique={name:ident for name,ident in rows};assert len(unique)==4,unique
 weights=ROOT/'vendor/salience_sgg/logs';weights.mkdir(parents=True,exist_ok=True)
 manifest=[]
 for name,ident in unique.items():
  path=weights/name;event('downloading_vg_asset',asset=name)
  if not path.exists():
   temp=path.with_suffix(path.suffix+'.part');r=gdown.download(id=ident,output=str(temp),quiet=True,resume=True)
   if not r:raise RuntimeError('public_weight_download_failed_'+name)
   temp.replace(path)
  with path.open('rb') as f:
   h=hashlib.sha256()
   for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
  manifest.append({'name':name,'public_file_id':ident,'bytes':path.stat().st_size,'sha256':h.hexdigest()})
 (ROOT/'reports/VG_V2_ASSETS.json').write_text(json.dumps({'upstream_commit':'d316a0b2057e549764eb8dd8a0bcce8b1136d98a','assets':manifest},indent=2))
 event('preparation_complete',assets=manifest,test_scored=False)
if __name__=='__main__':
 try:main()
 except Exception as e:
  event('preparation_failed',error_type=type(e).__name__,reason=str(e)[:300]);raise
