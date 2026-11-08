"""Run fixed external qualitative inference and save stdout artifacts locally."""
from pathlib import Path
import base64,hashlib,json,subprocess,sys
ROOT=Path(__file__).absolute().parents[1];OUT=ROOT/'outputs/context_1729_coco_oi_visualization_20261024';OUT.mkdir(parents=True,exist_ok=True)
prereg={'checkpoint_sha256':'29017a69d7cb6d76d68c834b5d540712775a9e0cf9cae13efd17d78a7b69292e','selection_source':'reports/VG_CONTEXT_EXTERNAL_VISUAL_PREFLIGHT_20261024.json','worker_sha256':hashlib.sha256((ROOT/'scripts/remote_context_external_visual_infer_v1.py').read_bytes()).hexdigest(),'policy':'Six fixed images, no GT or quality filtering, no tuning, no metric, zero server writes; VG vocabulary retained.'}
(OUT/'qualitative_protocol.json').write_text(json.dumps(prereg,indent=2)+'\n',encoding='utf8')
call=subprocess.run([sys.executable,'-X','utf8',str(ROOT/'scripts/ssh_project.py'),'run','scripts/remote_context_external_visual_infer_v1.py','--environment','project','--report',str((OUT/'remote_export.json').relative_to(ROOT))],cwd=ROOT,capture_output=True,text=True,encoding='utf8',timeout=230)
if call.returncode:print(call.stderr[-3000:]);raise SystemExit(call.returncode)
raw=(OUT/'remote_export.json').read_text(encoding='utf8')
payload=json.loads(raw[raw.index('{"status": "EXPORTED"'):]);assert payload['status']=='EXPORTED' and len(payload['samples'])==6
vocab=json.loads((ROOT/'outputs/context_1729_test_visualization_20261024/samples.json').read_text(encoding='utf8'))
payload.update(object_names=vocab['object_names'],predicate_names=vocab['predicate_names'])
for r in payload['samples']:
 data=base64.b64decode(r.pop('image_jpeg_base64'),validate=True);assert hashlib.sha256(data).hexdigest()==r['image_jpeg_sha256'];stem='vg_'+str(r['image_id'])
 (OUT/(stem+'_source.jpg')).write_bytes(data);(OUT/(stem+'_prediction.json')).write_text(json.dumps(r,indent=2)+'\n',encoding='utf8')
(OUT/'samples.json').write_text(json.dumps(payload,indent=2)+'\n',encoding='utf8')
print(json.dumps({k:payload[k] for k in ['status','new_inference','new_test_scoring','server_files_written','max_own_gpu_mib','peak_rss_bytes','wall_seconds']}))
