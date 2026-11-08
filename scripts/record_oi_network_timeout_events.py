"""Deduplicate verified source-network failures without another remote poll."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
root=Path(__file__).absolute().parents[1]
report=json.loads((root/'reports/OI_EXHAUSTED_SOURCE_AUDIT.json').read_text(encoding='utf8'))
phase=report['preregistration_sha256']
assert phase=='61717d618eaab7306f7d17c302efdef095943f41fcf0d37a9488eb3b905ec775'
assert not any(report[k] for k in ('fingerprint_journals_modified','benchmark_test_scored','live_v2_modified'))
assert report['additional_network_bytes']==0
digest=report['evidence_artifact_sha256']
assert len(digest)==64 and all(c in '0123456789abcdef' for c in digest)
raw=(root/'artifacts/caption_readiness'/('oi_exhausted_sources_'+digest+'.json')).read_bytes()
assert hashlib.sha256(raw).hexdigest()==digest
evidence=json.loads(raw)
assert evidence['records']==report['records'] and evidence['preregistration_sha256']==phase
affected=[row for row in report['records'] if any('timed out' in str(a[3]) for a in row['attempts'])]
assert all(len(row['attempts'])>=2 for row in affected)
state_path=root/'reports/VG_V2_MONITOR_STATE.json'
state=json.loads(state_path.read_text(encoding='utf8'))
key='oi_source_network_timeout:'+phase
events=[]
if affected and key not in state['announced']:
    state['announced'].append(key)
    events.append({'type':'anomaly','condition':'oi_test_source_network_timeout_after_retry_cap',
                   'source_phase_sha256':phase,'evidence_sha256':digest,
                   'affected_image_ids':[row['image_id'] for row in affected],
                   'failed_ids_preserved_in_denominator':True,'additional_network_bytes':0,
                   'next':'Continue registered remaining identities; preserve existing ledger, retry cap and failed IDs; any supplemental technical retry needs a separate bounded phase with provenance and budget admission'})
out={'utc':datetime.now(timezone.utc).isoformat(),'notify':bool(events),'events':events,
     'evidence_sha256_verified':digest,'additional_ssh_poll':False}
def atomic(path,obj):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(obj,indent=2),encoding='utf8');temporary.replace(path)
atomic(state_path,state)
atomic(root/'reports/OI_NETWORK_TIMEOUT_NEW_EVENTS.json',out)
if events:
    path=root/'reports/VG_V2_MONITOR_LAST_CHECK.json'
    last=json.loads(path.read_text(encoding='utf8'))
    last['events'].extend(events);last['notify']=True
    last['independent_audit_event_utc']=out['utc'];atomic(path,last)
print(json.dumps(out))
