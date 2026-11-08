"""Fetch the bounded read-only export without printing image bytes; save locally."""
from pathlib import Path
import base64,hashlib,json,subprocess,sys
ROOT=Path(__file__).absolute().parents[1]
OUT=ROOT/'outputs/context_1729_test_visualization_20261024'
OUT.mkdir(parents=True,exist_ok=True)
export=OUT/'remote_export.json'
call=subprocess.run([sys.executable,'-X','utf8',str(ROOT/'scripts/ssh_project.py'),'run',
                     'scripts/remote_export_context_test_visual_samples_v1.py','--environment','system',
                     '--report',str(export.relative_to(ROOT))],cwd=ROOT,capture_output=True,text=True,encoding='utf8',timeout=240)
if call.returncode:
 print(call.stderr[-2000:]);raise SystemExit(call.returncode)
payload=json.loads(export.read_text(encoding='utf8'));assert payload['status']=='EXPORTED'
for record in payload['samples']:
 data=base64.b64decode(record.pop('image_jpeg_base64'),validate=True)
 assert hashlib.sha256(data).hexdigest()==record['image_jpeg_sha256']
 stem='vg_'+str(record['image_id']);(OUT/(stem+'_source.jpg')).write_bytes(data)
 (OUT/(stem+'_prediction.json')).write_text(json.dumps(record,indent=2)+'\n',encoding='utf8')
(OUT/'samples.json').write_text(json.dumps(payload,indent=2)+'\n',encoding='utf8')
print(json.dumps({'status':'LOCAL_SAMPLES_SAVED','directory':str(OUT),'image_ids':[r['image_id'] for r in payload['samples']],
                  'sample_indices':payload['sample_indices'],'source_prediction_sha256':payload['prediction_artifact_sha256'],
                  'new_inference':False,'new_test_scoring':False}))
