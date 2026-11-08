"""Verify cleanup/migrations, active VG lock and images without scoring anything."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,os,re,subprocess
from PIL import Image
home=Path.home();shared=home/'shared-assets';project=home/'rsg-former'
def alias(x):return re.sub(r'/(home|users)/[^/\s]+',r'/\1/${ACCOUNT}',str(x).replace(str(home),'${REMOTE_HOME}').replace(home.name,'${ACCOUNT}'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
cleanup=json.loads((shared/'maintenance/MODEL_CLEANUP_RESULT_20261010.json').read_text());registry=json.loads((shared/'ASSET_REGISTRY.json').read_text())
checks=[]
for event in cleanup['events']:
    if event['action']!='removed':continue
    path=Path(event['path'].replace('${REMOTE_HOME}',str(home)));compat=event['compatibility_alias']
    ok=(path.is_symlink() and alias(path.resolve())==compat) if compat else not os.path.lexists(path)
    assert ok
    checks.append({'path':event['path'],'deleted_or_expected_alias':ok,'compatibility_alias':compat})
qwen=Path(registry['qwen14b']['path'].replace('${REMOTE_HOME}',str(home)))
idx=json.loads((qwen/'model.safetensors.index.json').read_text());shards=sorted(set(idx['weight_map'].values()))
assert len(shards)==8 and all((qwen/n).is_file() for n in shards)
assert sha(qwen/'model.safetensors.index.json')==cleanup['qwen_index_sha256']
clip=Path(registry['clip']['path'].replace('${REMOTE_HOME}',str(home)))
assert sha(clip/'pytorch_model.bin')==cleanup['clip_weight_sha256']
prereg=json.loads((project/'reports/VG_METHOD_PREREGISTRATION.json').read_text())
source_ok={rel:sha(project/rel)==expected for rel,expected in prereg['sources'].items()};assert all(source_ok.values())
config_ok=sha(project/'configs/vg_method_v1.json')==prereg['config_sha256'];assert config_ok
pid=int((project/'artifacts/vg_method_v1/pipeline.pid').read_text().strip());proc=Path('/proc')/str(pid)
active=proc.exists() and b'remote_vg_method.py' in (proc/'cmdline').read_bytes()
datasets={}
for name,item in registry['datasets'].items():
    root=Path(item['root'].replace('${REMOTE_HOME}',str(home)));manifest=json.loads((root/'test_manifest.json').read_text());refs=json.loads((root/'test_references_coco.json').read_text())
    images=manifest['images'];assert len({r['id'] for r in images})==len(images)
    assert {r['id'] for r in images}=={r['id'] for r in refs['images']}
    assert all(a['image_id'] in {r['id'] for r in images} for a in refs['annotations'])
    checked=0;bad=[];missing=[]
    for row in images:
        p=root/'images'/row['file_name']
        if not p.is_file():missing.append(row['file_name']);continue
        try:
            with Image.open(p) as img:img.verify()
            checked+=1
        except Exception as e:bad.append({'filename':row['file_name'],'error_type':type(e).__name__})
    assert len(missing)==item['missing_test_images'] and checked+len(bad)==item['available_test_images']
    datasets[name]={'test_images_expected':len(images),'images_header_verified':checked,'missing_images':len(missing),'invalid_image_headers':bad,'reference_captions':len(refs['annotations']),'full_test_ready':not missing and not bad,'verification_limit':'Pillow Image.verify structural/header validation; no pixel decode or evaluation.'}
def du(root,depth):
    p=subprocess.run(['du','-x','-B1',f'--max-depth={depth}',str(root)],capture_output=True,text=True,timeout=120)
    return {'returncode':p.returncode,'rows':sorted([{'path':alias(line.split('\t',1)[1]),'allocated_bytes':int(line.split('\t',1)[0])} for line in p.stdout.splitlines() if '\t' in line],key=lambda r:r['allocated_bytes'],reverse=True)}
result={'at_utc':datetime.now(timezone.utc).isoformat(),'status':'PASS_WITH_KNOWN_FLICKR_IMAGE_GAP','model_cleanup_checks':checks,'qwen_shards_present':8,'clip_weight_sha256_matches':True,'old_fair_batch_pid_absent':not (Path('/proc')/'2916729').exists(),'VG_locked_sources_match':source_ok,'VG_config_hash_matches':config_ok,'VG_pipeline_pid':pid,'VG_pipeline_active':active,'VG_zip_sizes':{n:(home/'sbir/hienlth/vg'/n).stat().st_size for n in ['images.zip','images2.zip']},'datasets':datasets,'home_sizes':du(home,1),'fair_sizes':du(home/'fair-project',2),'free_bytes':os.statvfs(home).f_bavail*os.statvfs(home).f_frsize,'model_loaded_or_test_scored':False}
(shared/'maintenance/SHARED_ASSETS_VERIFICATION_20261010.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
