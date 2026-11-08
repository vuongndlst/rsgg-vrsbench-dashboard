"""Deduplicate newly verified exact-pixel exclusions without another SSH poll."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
root=Path(__file__).absolute().parents[1]
report=json.loads((root/'reports/CAPTION_EXACT_PIXEL_COMPARISON_V3.json').read_text(encoding='utf8'))
assert report['script_sha256']=='283dc9ad51ba79b1d0e51b11623f311d5c00f48d93e54a7290633785337bad10'
assert not report['benchmark_test_scored'] and not report['training_manifest_modified'] and not report['live_v2_modified']
sha=report['evidence_artifact_sha256'];assert len(sha)==64 and all(c in '0123456789abcdef' for c in sha)
path=root/'artifacts/caption_readiness'/('exact_pixel_comparison_v3_'+sha+'.json')
raw=path.read_bytes();assert hashlib.sha256(raw).hexdigest()==sha
evidence=json.loads(raw)
assert evidence['retained_draft_ids_with_exact_protected_matches']==report['retained_draft_ids_with_exact_protected_matches']
state_path=root/'reports/VG_V2_MONITOR_STATE.json';state=json.loads(state_path.read_text(encoding='utf8'))
events=[]
for split,identities in report['retained_draft_ids_with_exact_protected_matches'].items():
    for identity in identities:
        key='exact_retained_draft_protected_match:'+split+':'+str(identity)
        if key in state['announced']:continue
        matches=[row for field in ('internal_matches','external_matches') for row in evidence[field]
                 if row['vg_id']==identity and row['canonical_split']==split and row['retained_in_current_draft']]
        assert matches,'no_verified_pair_for_new_retained_identity'
        state['announced'].append(key)
        events.append({'type':'anomaly','condition':'exact_protected_pixels_retained_in_provisional_draft',
                       'canonical_split':split,'vg_id':identity,'evidence_sha256':sha,'verified_matches':matches,
                       'next':'Mandatory exclusion in a new final clean release before fresh training; preserve old draft and live v2',
                       'active_v2_modified':False,'test_scored':False})
out={'utc':datetime.now(timezone.utc).isoformat(),'notify':bool(events),'events':events,
     'evidence_sha256_verified':sha,'additional_ssh_poll':False}
def atomic(p,obj):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf8');t.replace(p)
atomic(state_path,state)
atomic(root/'reports/VG_EXACT_PIXEL_NEW_EVENTS.json',out)
if events:
    last_path=root/'reports/VG_V2_MONITOR_LAST_CHECK.json';last=json.loads(last_path.read_text(encoding='utf8'))
    last['events'].extend(events);last['notify']=True
    last['independent_audit_event_utc']=out['utc'];atomic(last_path,last)
print(json.dumps(out,ensure_ascii=False))
