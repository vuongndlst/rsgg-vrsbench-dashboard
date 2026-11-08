"""Read-only diagnosis after a blocked batch; never process queries or edit journals."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, os, resource, shutil, signal, sqlite3
os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
resource.setrlimit(resource.RLIMIT_AS, (768*1024**2, 768*1024**2))
signal.alarm(60)
root=Path.home()/'rsg-former';base=root/'artifacts/caption_readiness'
checks={};observed={}
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return h.hexdigest()
observed['disk_free_bytes']=shutil.disk_usage(root).free
checks['disk_reserve_plus_private_journal']=observed['disk_free_bytes']>16*1024**3+128*1024**2
expected={'scripts/remote_vg_internal_phash_batch_v1.py':'20506f9ed3447644f8bd12cd568403b2fedf783d75e2a249763a3486cb856a62','src/rsgformer/caption_phash_candidates_v1.py':'827ac8e5a8ce4b0c8921b606b7feadf33c9df5f721f39f2a5f659f8b532f6688','configs/caption_perceptual_exclusion_v1.json':'f3efd67929926cbad2c76715d0ae768fa750dead3af41c81033d4951f0e9195b','scripts/remote_vg_pixel_fingerprint_batch.py':'5573148955570b19bc9f8e3c6834462c4ac9e2709f299e5b1ba3cc74ea48622c'}
for name,value in expected.items():
 observed[name]=digest(root/name);checks[name]=observed[name]==value
observed['vg_index_report_status']=json.loads((root/'reports/VG_PIXEL_FINGERPRINT_PROGRESS.json').read_text())['status']
checks['vg_index_report_complete']=observed['vg_index_report_status']=='VG_PIXEL_INDEX_COMPLETE'
source_path=base/'vg_pixel_fingerprints_v1.sqlite'
observed['original_vg_database_sha256']=digest(source_path)
checks['original_vg_database_unchanged']=observed['original_vg_database_sha256']=='983789d6425c628ac22849090f6160f8b3e55910da4f1aecad8a5b61a288e6f9'
with sqlite3.connect(source_path.as_uri()+'?mode=ro',uri=True,timeout=5) as db:
 db.execute('PRAGMA query_only=ON');db.execute('BEGIN')
 observed['vg_split_counts']=dict(db.execute('SELECT split,COUNT(*) FROM pixels GROUP BY split').fetchall())
 observed['vg_provenance']=json.loads(db.execute('SELECT value FROM provenance WHERE key="source"').fetchone()[0])
checks['vg_split_counts']=observed['vg_split_counts']=={'train':57723,'val':5000,'test':26446}
observed['draft_manifest_sha256']=digest(base/'canonical_internal_external_identity_splits_v2.json')
checks['draft_unchanged']=observed['draft_manifest_sha256']=='ade090fd4975e12e1ade2ac16a734f7b167f6630b38d1710cefa4176523f7b16'
observed['internal_preregistration_sha256']=digest(base/'vg_internal_phash_candidates_v1/preregistration.json')
checks['internal_preregistration_unchanged']=observed['internal_preregistration_sha256']=='c1c106f784e294af6181c7d0429098bf928041331dc2cc501e369c802e3792ab'
print(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),'status':'PASS' if all(checks.values()) else 'BLOCKED_PRECONDITION_CONFIRMED','checks':checks,'observed':observed,'journal_mutations':False,'candidate_queries_executed':0,'test_scored':False}))
