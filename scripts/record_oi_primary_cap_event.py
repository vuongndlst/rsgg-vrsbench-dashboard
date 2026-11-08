"""Record new independently diagnosed gate once; no extra SSH monitor call."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json
root=Path(__file__).absolute().parents[1]
headers=json.loads((root/'reports/OI_FAILED_HEADER_PROBE_V3.json').read_text())
assert headers['status']=='HEADER_AUDIT_COMPLETE'
assert headers['preregistration_sha256']=='daefdcc56989418d1f478ef93f2371f62af5347a94e7e29bd480d042d85c6f48'
assert not headers['full_image_downloaded'] and not headers['benchmark_test_scored']
assert hashlib.sha256((root/'src/rsgformer/openimages_test_reader_v2.py').read_bytes()).hexdigest()=='8f484a9b98683102a77f0cb4a3d85fb246a6670e63f7b6841261a00b9e915555'
blocked=[{'image_id':row['image_id'],'dimensions':row['header_dimensions'],
          'primary_pixels':row['header_dimensions'][0]*row['header_dimensions'][1]}
         for row in headers['records'] if row['header_dimensions'][0]*row['header_dimensions'][1]>16000000]
assert len(blocked)==2
event={'type':'resource_gate_failed','condition':'oi_sources_exceed_registered_primary_frame_pixel_cap',
       'stage':'openimages_full_test_pixel_identity_audit','primary_frame_pixel_cap':16000000,
       'blocked_sources':blocked,'header_preregistration_sha256':headers['preregistration_sha256'],
       'full_body_retry_launched':False,'test_scored':False,'live_v2_modified':False,
       'next':'Preserve all IDs; current reader cannot certify these sources within its cap. Continue independent work; no silent cap increase or missing-ID omission'}
key='oi_source_primary_pixel_cap:reader_v2:16000000'
state_path=root/'reports/VG_V2_MONITOR_STATE.json'
state=json.loads(state_path.read_text())
new=key not in state['announced']
if new:
    state['announced'].append(key)
    state_path.write_text(json.dumps(state,indent=2),encoding='utf8')
    check_path=root/'reports/VG_V2_MONITOR_LAST_CHECK.json'
    check=json.loads(check_path.read_text())
    check['events'].append(event)
    check.update(notify=True,independent_preparation_event_added_utc=datetime.now(timezone.utc).isoformat(),ssh_monitor_rerun=False)
    check_path.write_text(json.dumps(check,indent=2),encoding='utf8')
out={'utc':datetime.now(timezone.utc).isoformat(),'notify':new,'event':event,
     'reader_sha_verified':True,'real_training_state_modified':False}
(root/'reports/OI_PRIMARY_CAP_EVENT.json').write_text(json.dumps(out,indent=2),encoding='utf8')
print(json.dumps(out))
