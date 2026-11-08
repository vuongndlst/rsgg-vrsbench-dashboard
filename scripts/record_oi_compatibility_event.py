"""Persist today's independent audited event without repeating the SSH monitor."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json
root=Path(__file__).absolute().parents[1]
report=json.loads((root/'reports/OI_EXHAUSTED_SOURCE_AUDIT.json').read_text())
repair=json.loads((root/'reports/OI_MPO_REPAIR_PILOT.json').read_text())
assert report['retry_exhausted_images']==2 and repair['status']=='PASS' and repair['indexed_images']==2
evidence=root/'artifacts/caption_readiness/oi_mpo_repair_pilot_v1/pixel_fingerprints.json'
assert hashlib.sha256(evidence.read_bytes()).hexdigest()==repair['evidence_artifact_sha256']
key='oi_full_stream_retry_exhausted:'+report['preregistration_sha256']
state_path=root/'reports/VG_V2_MONITOR_STATE.json'
state=json.loads(state_path.read_text())
new=key not in state['announced']
event={'type':'anomaly','condition':'oi_test_pixel_sources_exhausted_registered_attempts',
       'stage':'openimages_full_test_pixel_identity_audit','missing_ids_at_first_notification':2,
       'diagnosis':'official .jpg objects are MPO,2frames,primary3648x2736/4000x3000; original JPEG-only/1024 assumption too strict',
       'separate_reader_v2_actual_pilot':'PASS2of2/17.49s/9265785bytes',
       'old_reader_or_journal_changed':False,'vg_job_modified':False,
       'supplemental_fingerprints_joined':False,'test_scored':False,
       'next':'Complete separately versioned supplemental union for all failed sources, preserve every denominator ID and old phase caps/outcome'}
if new:
    state['announced'].append(key)
    state_path.write_text(json.dumps(state,indent=2),encoding='utf8')
    check_path=root/'reports/VG_V2_MONITOR_LAST_CHECK.json'
    check=json.loads(check_path.read_text())
    check['events'].append(event)
    check.update(notify=True,independent_preparation_event_added_utc=datetime.now(timezone.utc).isoformat(),ssh_monitor_rerun=False)
    check_path.write_text(json.dumps(check,indent=2),encoding='utf8')
out={'utc':datetime.now(timezone.utc).isoformat(),'notify':new,'event':event,
     'immutable_supplemental_evidence_sha_verified':True,'real_training_state_modified':False}
(root/'reports/OI_COMPATIBILITY_EVENT.json').write_text(json.dumps(out,indent=2),encoding='utf8')
print(json.dumps(out))
