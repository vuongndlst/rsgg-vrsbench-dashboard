"""Read-only qualitative export of six fixed samples from saved Context TEST outputs.

No inference, evaluator, model selection, extraction to server disk or metric access.
"""
from pathlib import Path
from datetime import datetime, timezone
import base64,gzip,hashlib,json,os,random,resource,signal,time,zipfile
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
resource.setrlimit(resource.RLIMIT_AS,(1536*1024**2,1536*1024**2))
resource.setrlimit(resource.RLIMIT_CPU,(120,120));signal.alarm(150)
root=Path.home()/'rsg-former';art=root/'artifacts/vg_component_control_test_v1';started=time.monotonic()
result=json.loads((art/'result.json').read_text());lock=json.loads((art/'preregistration.json').read_text())
assert result['status']=='COMPONENT_TEST_COMPLETE' and result['variant']=='context' and result['seed']==1729 and result['selected_epoch']==10
assert result['checkpoint_sha256']==lock['checkpoint_sha256']=='29017a69d7cb6d76d68c834b5d540712775a9e0cf9cae13efd17d78a7b69292e'
assert hashlib.sha256((art/'preregistration.json').read_bytes()).hexdigest()=='6b57ddbdbaf51c7d9307ae33f9766efc74cfe639cc51b98f9e36adf9dd659b77'
predictions=art/'test_predictions.jsonl.gz';h=hashlib.sha256()
with predictions.open('rb') as stream:
 for block in iter(lambda:stream.read(4*1024**2),b''):h.update(block)
assert h.hexdigest()==result['artifact_hashes'][predictions.name]
with zipfile.ZipFile(root/'data/vg_canonical_audit/reltr_vg_annotations.zip') as z:
 data=json.loads(z.read('vg/test.json'))
 info={int(x['id']):{'id':int(x['id']),'width':x['width'],'height':x['height']} for x in data['images']}
 objects={str(x['id']):x['name'] for x in data['categories'] if int(x['id'])>0};del data
 relation_data=json.loads(z.read('vg/rel.json'));predicates=relation_data['rel_categories'];del relation_data
ids=sorted(info);assert len(ids)==26446
indices=sorted(random.Random(1729).sample(range(len(ids)),6));wanted=set(indices);records=[]
with gzip.open(predictions,'rb') as stream:
 for i,line in enumerate(stream):
  if i in wanted:
   record=json.loads(line);assert record['image_id']==ids[i];record['test_index']=i;records.append(record)
  if i==max(indices):break
assert len(records)==6
archives=[zipfile.ZipFile(Path.home()/'sbir/hienlth/vg'/name) for name in ['images.zip','images2.zip']]
members={}
for z in archives:
 for m in z.infolist():
  if m.filename.lower().endswith('.jpg'):
   try:identity=int(Path(m.filename).stem)
   except ValueError:continue
   if identity in {r['image_id'] for r in records}:assert identity not in members;members[identity]=(z,m)
for r in records:
 z,m=members[r['image_id']];assert m.file_size<=16*1024**2
 raw=z.read(m);r.update(image=info[r['image_id']],image_jpeg_base64=base64.b64encode(raw).decode(),image_jpeg_sha256=hashlib.sha256(raw).hexdigest())
for z in archives:z.close()
out={'utc':datetime.now(timezone.utc).isoformat(),'status':'EXPORTED','source':'saved single-pass Context TEST predictions; no new inference or scoring',
     'checkpoint_sha256':lock['checkpoint_sha256'],'selected_epoch':10,'seed':1729,'variant':'context',
     'prediction_artifact_sha256':h.hexdigest(),'sample_selection':'random.Random(1729).sample(range(26446),6); sorted indices; no correctness/score filtering',
     'sample_indices':indices,'object_names':objects,'predicate_names':predicates,'samples':records,
     'server_files_written':0,'new_model_inference':False,'new_test_scoring':False,'wall_seconds':time.monotonic()-started}
print(json.dumps(out))
