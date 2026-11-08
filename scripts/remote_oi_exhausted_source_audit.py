"""Read-only failure evidence from the registered identity audit journal."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, os, signal, sqlite3
root=Path.home()/'rsg-former'
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
signal.alarm(15)
path=root/'artifacts/caption_readiness/oi_stream_fingerprints_full_v1.sqlite'
db=sqlite3.connect(path.as_uri()+'?mode=ro',uri=True,timeout=5)
try:
    db.execute('PRAGMA query_only=ON')
    db.execute('BEGIN')
    source=json.loads(db.execute('SELECT value FROM provenance WHERE key=?',('source',)).fetchone()[0])
    assert source['script_sha256']=='752b93fc302851e5e1922bf0a918842f09fca29eade1136cf2b65a4d69f0e1dd'
    assert source['preregistration_sha256']=='61717d618eaab7306f7d17c302efdef095943f41fcf0d37a9488eb3b905ec775'
    identities=[row[0] for row in db.execute('SELECT image_id FROM attempts GROUP BY image_id HAVING COUNT(*)>=2 AND NOT EXISTS(SELECT 1 FROM pixels p WHERE p.image_id=attempts.image_id) ORDER BY image_id')]
    assert len(identities)<=4096, 'bounded_failure_evidence_limit'
    records=[]
    for identity in identities:
        rows=db.execute('SELECT state,charged_bytes,actual_network_bytes,reason,utc FROM attempts WHERE image_id=? ORDER BY seq',(identity,)).fetchall()
        records.append({'image_id':identity,'attempts':rows})
    report={'utc':datetime.now(timezone.utc).isoformat(),
            'status':'RETRY_EXHAUSTED_TEST_SOURCES' if identities else 'NO_RETRY_EXHAUSTED_SOURCES',
            'preregistration_sha256':source['preregistration_sha256'],
            'retry_exhausted_images':len(identities),'records':records,
            'fingerprint_journals_modified':False,'additional_network_bytes':0,
            'benchmark_test_scored':False,'live_v2_modified':False,
            'next':'Preserve missing IDs and denominator; continue registered remaining identities; technical repair requires separate version/budget after exact phase preservation, never silently relax decoder or attempt cap'}
    raw=json.dumps(report,sort_keys=True,separators=(',',':')).encode()
    evidence_sha=hashlib.sha256(raw).hexdigest()
    folder=root/'artifacts/caption_readiness'
    evidence=folder/('oi_exhausted_sources_'+evidence_sha+'.json')
    with evidence.open('xb') as stream:stream.write(raw)
    report['evidence_artifact_sha256']=evidence_sha
    (root/'reports/OI_EXHAUSTED_SOURCE_AUDIT.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report))
finally:
    db.close()
